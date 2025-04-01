package com.example.lanedetectionapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.os.Build
import android.util.Log
import android.util.Size
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.lanedetectionapp.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import androidx.appcompat.app.AlertDialog

@OptIn(androidx.camera.core.ExperimentalGetImage::class)
class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"
    private lateinit var binding: ActivityMainBinding

    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    private var isDetectionRunning = false
    private lateinit var laneDetector: LaneDetector

    // Surface for drawing lanes
    private lateinit var laneOverlaySurface: SurfaceView
    private var laneSurfaceHolder: SurfaceHolder? = null
    private var lanePaint: Paint? = null

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

        // Initialize surface for lane drawing
        laneOverlaySurface = binding.laneOverlaySurface
        laneSurfaceHolder = laneOverlaySurface.holder

        // Setup transparent background for the overlay
        laneOverlaySurface.setZOrderOnTop(true)
        laneSurfaceHolder?.setFormat(PixelFormat.TRANSPARENT)

        // Initialize paint for lane drawing
        lanePaint = Paint().apply {
            isAntiAlias = true
            style = Paint.Style.STROKE
            strokeWidth = 5f
        }

        // Check permissions and start camera
        checkPermissionsAndStartCamera()

        // Set up the buttons
        binding.toggleDetectionButton.setOnClickListener {
            isDetectionRunning = !isDetectionRunning
            binding.toggleDetectionButton.text = if (isDetectionRunning) "Stop Detection" else "Start Detection"

            // Clear lane overlay when detection is stopped
            if (!isDetectionRunning) {
                clearLaneOverlay()
            }
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

    private fun startCamera() {
        Log.d(TAG, "Starting Camera Setup")
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Set up the preview use case
            val preview = Preview.Builder()
                .setTargetResolution(Size(400, 144))  // Match with model input size
                .build()
                .also {
                    it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
                }

            // Set up the image capture use case
            imageCapture = ImageCapture.Builder()
                .setTargetResolution(Size(400, 144))  // Match with model input size
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            // Set up the image analysis use case
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(400, 144))  // Match with model input size
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST) // Only process latest frame
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888) // Specify format
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        if (isDetectionRunning) {
                            processImageProxy(imageProxy)
                        } else {
                            // Important: close the image if not processing it
                            try {
                                imageProxy.close()
                            } catch (e: Exception) {
                                Log.e(TAG, "Error closing image proxy: ${e.message}")
                            }
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
        try {
            val image = imageProxy.image ?: run {
                imageProxy.close()
                return
            }

            // Convert the image to bitmap
            val bitmap = ImageUtils.imageProxyToBitmap(imageProxy)

            if (bitmap != null) {
                try {
                    // Detect lanes and get lane points rather than a new bitmap
                    val (lanePoints, inferenceTime) = laneDetector.detectLanesPoints(bitmap)

                    // Update the UI with the inference time
                    runOnUiThread {
                        binding.inferenceTimeTextView.text = "Inference Time: $inferenceTime ms"
                    }

                    // Draw the lane points on the overlay surface
                    drawLanesOnSurface(lanePoints)

                    // Safely recycle the bitmap
                    if (!bitmap.isRecycled) {
                        bitmap.recycle()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error in lane detection process: ${e.message}")
                    e.printStackTrace()

                    // Make sure to recycle the bitmap in case of error
                    if (!bitmap.isRecycled) {
                        bitmap.recycle()
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing image: ${e.message}")
            e.printStackTrace()
        } finally {
            // Always close the imageProxy to release resources
            try {
                imageProxy.close()
            } catch (e: Exception) {
                Log.e(TAG, "Error closing image proxy: ${e.message}")
            }
        }
    }

    private fun drawLanesOnSurface(lanePoints: Array<List<Pair<Float, Float>>>) {
        // Get the canvas from the surface holder
        val canvas = laneSurfaceHolder?.lockCanvas()
        canvas?.let {
            try {
                // Clear the canvas
                it.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

                // Scale factors to map from model dimensions to view dimensions
                val scaleX = laneOverlaySurface.width.toFloat() / laneDetector.getInputWidth()
                val scaleY = laneOverlaySurface.height.toFloat() / laneDetector.getInputHeight()

                // Draw each lane
                val laneColors = arrayOf(Color.GREEN, Color.BLUE)

                for (laneIdx in lanePoints.indices) {
                    val lane = lanePoints[laneIdx]
                    if (lane.size < 2) continue

                    // Set lane color
                    lanePaint?.color = laneColors[laneIdx % laneColors.size]

                    // Create a path for the lane
                    val path = Path()
                    val sortedPoints = lane.sortedBy { it.second }

                    // Move to first point
                    path.moveTo(sortedPoints[0].first * scaleX, sortedPoints[0].second * scaleY)

                    // Add line segments to subsequent points
                    for (i in 1 until sortedPoints.size) {
                        path.lineTo(sortedPoints[i].first * scaleX, sortedPoints[i].second * scaleY)
                    }

                    // Draw the path
                    lanePaint?.let { paint ->
                        it.drawPath(path, paint)
                    }

                    // Optionally draw points for visualization
                    val pointPaint = Paint().apply {
                        color = laneColors[laneIdx % laneColors.size]
                        style = Paint.Style.FILL
                        isAntiAlias = true
                    }

                    for (point in sortedPoints) {
                        it.drawCircle(point.first * scaleX, point.second * scaleY, 3f, pointPaint)
                    }
                }

                // Add inference time info
                val textPaint = Paint().apply {
                    color = Color.WHITE
                    textSize = 30f
                    style = Paint.Style.FILL
                    setShadowLayer(2f, 1f, 1f, Color.BLACK)
                }

            } finally {
                // Release the canvas
                laneSurfaceHolder?.unlockCanvasAndPost(it)
            }
        }
    }

    private fun clearLaneOverlay() {
        val canvas = laneSurfaceHolder?.lockCanvas()
        canvas?.let {
            try {
                // Clear the canvas
                it.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
            } finally {
                laneSurfaceHolder?.unlockCanvasAndPost(it)
            }
        }
    }

    private fun takePhoto() {
        val imageCapture = imageCapture ?: return

        try {
            // Use the image capture for a single photo
            imageCapture.takePicture(
                ContextCompat.getMainExecutor(this),
                object : ImageCapture.OnImageCapturedCallback() {
                    @ExperimentalGetImage
                    override fun onCaptureSuccess(image: ImageProxy) {
                        try {
                            // Convert captured image to bitmap
                            val bitmap = ImageUtils.imageProxyToBitmap(image)

                            if (bitmap != null) {
                                try {
                                    // Process the bitmap with lane detection
                                    val (lanePoints, inferenceTime) = laneDetector.detectLanesPoints(bitmap)

                                    // Update the inference time display
                                    runOnUiThread {
                                        binding.inferenceTimeTextView.text = "Inference Time: $inferenceTime ms"
                                    }

                                    // Draw the lanes on the overlay
                                    drawLanesOnSurface(lanePoints)

                                    // Clean up
                                    if (!bitmap.isRecycled) {
                                        bitmap.recycle()
                                    }

                                    Toast.makeText(baseContext, "Photo captured", Toast.LENGTH_SHORT).show()
                                } catch (e: Exception) {
                                    Log.e(TAG, "Error processing capture: ${e.message}")
                                    e.printStackTrace()

                                    // Clean up bitmap on error
                                    if (!bitmap.isRecycled) {
                                        bitmap.recycle()
                                    }
                                }
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Error processing captured photo: ${e.message}")
                            e.printStackTrace()
                        } finally {
                            // Important: close the image
                            try {
                                image.close()
                            } catch (e: Exception) {
                                Log.e(TAG, "Error closing captured image: ${e.message}")
                            }
                        }
                    }

                    override fun onError(exception: ImageCaptureException) {
                        Log.e(TAG, "Photo capture failed: ${exception.message}", exception)
                        Toast.makeText(baseContext, "Photo capture failed: ${exception.message}",
                            Toast.LENGTH_SHORT).show()
                    }
                }
            )
        } catch (e: Exception) {
            // Fallback to show error
            Log.e(TAG, "Error taking photo: ${e.message}")
            Toast.makeText(baseContext, "Error taking photo: ${e.message}",
                Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Ensure we shut down executors and close resources
        cameraExecutor.shutdown()
        laneDetector.close()
    }
}