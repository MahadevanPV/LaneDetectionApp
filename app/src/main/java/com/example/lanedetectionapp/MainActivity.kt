package com.example.lanedetectionapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Bundle
import android.os.Build
import android.util.Log
import android.util.Size
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.lanedetectionapp.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import com.example.lanedetectionapp.LaneDetector
import com.example.lanedetectionapp.ImageUtils
import androidx.appcompat.app.AlertDialog

@OptIn(androidx.camera.core.ExperimentalGetImage::class)
class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"
    private lateinit var binding: ActivityMainBinding

    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    private var isDetectionRunning = false
    private lateinit var laneDetector: LaneDetector

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS =
            if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                arrayOf(
                    Manifest.permission.CAMERA,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
                )
            } else {
                arrayOf(Manifest.permission.CAMERA)
            }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize lane detector
        laneDetector = LaneDetector(this)

        // Configure the overlay image view scale type
        binding.overlayImageView.scaleType = ImageView.ScaleType.FIT_XY

        // Check permissions and start camera
        checkPermissionsAndStartCamera()

        // Set up the buttons
        binding.toggleDetectionButton.setOnClickListener {
            isDetectionRunning = !isDetectionRunning
            binding.toggleDetectionButton.text = if (isDetectionRunning) "Stop Detection" else "Start Detection"
        }

        binding.captureButton.setOnClickListener {
            takePhoto()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun checkPermissionsAndStartCamera() {
        if (allPermissionsGranted()) {
            Log.d(TAG, "All permissions already granted")
            startCamera()
        } else {
            Log.d(TAG, "Requesting permissions")
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        val isGranted = ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        Log.d(TAG, "Permission check for $it: $isGranted")
        isGranted
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            // Log detailed permission results
            for (i in permissions.indices) {
                val permission = permissions[i]
                val granted = grantResults.getOrNull(i) == PackageManager.PERMISSION_GRANTED
                Log.d(TAG, "Permission result: $permission = $granted")
            }

            if (allPermissionsGranted()) {
                Log.d(TAG, "All permissions granted in callback")
                startCamera()
            } else {
                Log.e(TAG, "Permissions denied")
                Toast.makeText(
                    this,
                    "Permission not granted. The app needs camera access to function properly.",
                    Toast.LENGTH_LONG
                ).show()

                // Give the user another chance with an explanation
                if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
                    showPermissionRationaleDialog()
                }
            }
        }
    }

    private fun showPermissionRationaleDialog() {
        AlertDialog.Builder(this)
            .setTitle("Camera Permission Required")
            .setMessage("This app needs camera access to detect lanes. Please grant the permission.")
            .setPositiveButton("Grant") { _, _ ->
                ActivityCompat.requestPermissions(
                    this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
                )
            }
            .setNegativeButton("Cancel") { dialog, _ ->
                dialog.dismiss()
                Toast.makeText(this, "App cannot function without camera permission", Toast.LENGTH_LONG).show()
            }
            .show()
    }

    private fun requestPermissions() {
        ActivityCompat.requestPermissions(
            this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
        )
    }

    private fun startCamera() {
        Log.d(TAG, "Starting Camera Setup")
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Set up the preview use case
            val preview = Preview.Builder()
                .setTargetResolution(Size(640, 360))  // Match with model input size
                .build()
                .also {
                    it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
                }

            // Set up the image capture use case
            imageCapture = ImageCapture.Builder()
                .setTargetResolution(Size(640, 360))  // Match with model input size
                .build()

            // Set up the image analysis use case
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 360))  // Match with model input size
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        if (isDetectionRunning) {
                            processImageProxy(imageProxy)
                        } else {
                            imageProxy.close()
                        }
                    }
                }

            // Select back camera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind any bound use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer)

                // Add success log
                Log.d(TAG, "Camera setup complete")

            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImageProxy(imageProxy: ImageProxy) {
        val image = imageProxy.image ?: run {
            imageProxy.close()
            return
        }

        try {
            // Convert the image to bitmap properly
            val bitmap = imageProxyToBitmap(imageProxy)
            if (bitmap != null) {
                // Detect lanes on the bitmap
                val (resultBitmap, inferenceTime) = laneDetector.detectLanes(bitmap)

                // Update the overlay image view on the main thread
                runOnUiThread {
                    try {
                        // Create a copy before setting it to ImageView and recycling the original
                        val displayBitmap = resultBitmap.copy(resultBitmap.config, false)
                        binding.overlayImageView.setImageBitmap(displayBitmap)
                        binding.inferenceTimeTextView.text = "Inference Time: $inferenceTime ms"
                    } catch (e: Exception) {
                        Log.e(TAG, "Error displaying result: ${e.message}")
                    } finally {
                        // Now it's safe to recycle the original
                        if (!resultBitmap.isRecycled) {
                            resultBitmap.recycle()
                        }
                    }
                }

                // Safely recycle the original bitmap
                if (!bitmap.isRecycled) {
                    bitmap.recycle()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing image: ${e.message}")
        } finally {
            imageProxy.close()
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return ImageUtils.imageProxyToBitmap(imageProxy)
    }

    private fun takePhoto() {
        val imageCapture = imageCapture ?: return

        try {
            // For demonstration, try using the actual camera capture instead of a dummy bitmap
            imageCapture.takePicture(
                ContextCompat.getMainExecutor(this),
                object : ImageCapture.OnImageCapturedCallback() {
                    @ExperimentalGetImage
                    override fun onCaptureSuccess(image: ImageProxy) {
                        try {
                            // Convert captured image to bitmap
                            val bitmap = imageProxyToBitmap(image)
                            if (bitmap != null) {
                                // Process the bitmap with lane detection
                                val (resultBitmap, inferenceTime) = laneDetector.detectLanes(bitmap)

                                // Display the result
                                val displayBitmap = resultBitmap.copy(resultBitmap.config, false)
                                binding.overlayImageView.setImageBitmap(displayBitmap)
                                binding.inferenceTimeTextView.text = "Inference Time: $inferenceTime ms"

                                // Clean up
                                if (!resultBitmap.isRecycled) {
                                    resultBitmap.recycle()
                                }
                                if (!bitmap.isRecycled) {
                                    bitmap.recycle()
                                }

                                Toast.makeText(baseContext, "Photo captured", Toast.LENGTH_SHORT).show()
                            }

                            // Close the image proxy
                            image.close()
                        } catch (e: Exception) {
                            Log.e(TAG, "Error processing captured photo: ${e.message}")
                            image.close()
                        }
                    }

                    override fun onError(exception: ImageCaptureException) {
                        Log.e(TAG, "Photo capture failed: ${exception.message}", exception)
                        Toast.makeText(baseContext, "Photo capture failed", Toast.LENGTH_SHORT).show()
                    }
                }
            )
        } catch (e: Exception) {
            // Fallback to using a resource bitmap if camera capture fails
            try {
                val bitmap = BitmapFactory.decodeResource(resources, android.R.drawable.ic_menu_camera)

                // Process the bitmap
                val (resultBitmap, inferenceTime) = laneDetector.detectLanes(bitmap)

                try {
                    // Display the result - use a copy to prevent recycling issues
                    val displayBitmap = resultBitmap.copy(resultBitmap.config, false)
                    binding.overlayImageView.setImageBitmap(displayBitmap)
                    binding.inferenceTimeTextView.text = "Inference Time: $inferenceTime ms"

                    Toast.makeText(baseContext, "Photo captured (fallback)", Toast.LENGTH_SHORT).show()
                } finally {
                    // Recycle bitmaps we created
                    if (!resultBitmap.isRecycled) {
                        resultBitmap.recycle()
                    }
                }
            } catch (e2: Exception) {
                Log.e(TAG, "Error processing photo (fallback): ${e2.message}")
                Toast.makeText(baseContext, "Error processing photo", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        laneDetector.close()
    }
}