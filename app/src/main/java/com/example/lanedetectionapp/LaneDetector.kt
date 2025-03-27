package com.example.lanedetectionapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.pow

class LaneDetector(private val context: Context) {
    private val TAG = "LaneDetector"
    private var interpreter: Interpreter? = null

    // Define model parameters based on your model details
    private val modelName = "polylanenet_basic.tflite"
    private val inputImageWidth = 640  // Actual model input size from your test
    private val inputImageHeight = 360 // Actual model input size from your test
    private val numChannels = 3  // RGB

    // Performance tracking
    private var inferenceTime: Long = 0

    init {
        try {
            // Load the TFLite model
            val tfliteModel = loadModelFile()
            val options = Interpreter.Options()
            // Set num threads for better performance on multi-core devices
            options.setNumThreads(4)
            // Disable NNAPI as it might not support all ops
            options.setUseNNAPI(false)
            interpreter = Interpreter(tfliteModel, options)
            Log.d(TAG, "TFLite model loaded successfully")

            // Log model input and output details for debugging
            if (interpreter != null) {
                val inputTensor = interpreter!!.getInputTensor(0)
                val outputTensor = interpreter!!.getOutputTensor(0)
                Log.d(TAG, "Model input shape: ${inputTensor.shape().contentToString()}")
                Log.d(TAG, "Model output shape: ${outputTensor.shape().contentToString()}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading TFLite model: ${e.message}")
            // Don't throw an exception here - just log the error
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        // Resize the bitmap to match model input size
        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmap, inputImageWidth, inputImageHeight, true)

        // Allocate ByteBuffer for model input
        val modelInputSize = inputImageWidth * inputImageHeight * numChannels * 4 // 4 bytes per float
        val inputBuffer = ByteBuffer.allocateDirect(modelInputSize)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Convert bitmap to normalized float values between 0-1
        for (y in 0 until inputImageHeight) {
            for (x in 0 until inputImageWidth) {
                val pixel = resizedBitmap.getPixel(x, y)
                // Extract RGB values (0-255)
                val r = Color.red(pixel)
                val g = Color.green(pixel)
                val b = Color.blue(pixel)

                // Normalize to 0-1 range
                inputBuffer.putFloat(r / 255.0f)
                inputBuffer.putFloat(g / 255.0f)
                inputBuffer.putFloat(b / 255.0f)
            }
        }

        inputBuffer.rewind()
        return inputBuffer
    }

    fun detectLanes(inputBitmap: Bitmap): Pair<Bitmap, Long> {
        if (interpreter == null) {
            Log.e(TAG, "Interpreter is not initialized")
            // Create a copy of the input bitmap instead of returning the original
            val resultBitmap = inputBitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(resultBitmap)
            val paint = Paint().apply {
                color = Color.RED
                textSize = 50f
                style = Paint.Style.FILL
            }
            // Draw text indicating model failed to load
            canvas.drawText("Model not loaded", 50f, 100f, paint)
            return Pair(resultBitmap, 0L)
        }

        val startTime = SystemClock.uptimeMillis()

        try {
            // Prepare input data
            val inputBuffer = preprocess(inputBitmap)

            // Get the output tensor shape - based on your test data: [1, 5, 7]
            val numLanes = 5
            val numCoefficients = 7

            // Create output array with the correct shape
            val outputArray = Array(1) { Array(numLanes) { FloatArray(numCoefficients) } }

            // Run inference
            interpreter!!.run(inputBuffer, outputArray)

            inferenceTime = SystemClock.uptimeMillis() - startTime
            Log.d(TAG, "Inference completed in $inferenceTime ms")

            // Process output to draw lanes on the bitmap
            val resultBitmap = drawLanesFromPolynomials(inputBitmap, outputArray[0])

            return Pair(resultBitmap, inferenceTime)
        } catch (e: Exception) {
            Log.e(TAG, "Error during inference: ${e.message}")
            e.printStackTrace()
            // Create a fallback result
            val resultBitmap = inputBitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(resultBitmap)
            val paint = Paint().apply {
                color = Color.RED
                textSize = 40f
                style = Paint.Style.FILL
            }
            canvas.drawText("Inference error: ${e.message}", 50f, 100f, paint)

            inferenceTime = SystemClock.uptimeMillis() - startTime
            return Pair(resultBitmap, inferenceTime)
        }
    }

    private fun drawLanesFromPolynomials(
        originalBitmap: Bitmap,
        laneCoefficients: Array<FloatArray>
    ): Bitmap {
        // Create a mutable copy of the original bitmap
        val resultBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(resultBitmap)

        // Define lane colors
        val laneColors = arrayOf(
            Color.GREEN, Color.BLUE, Color.YELLOW, Color.CYAN, Color.MAGENTA
        )

        try {
            // Get bitmap dimensions for scaling
            val bmpWidth = resultBitmap.width
            val bmpHeight = resultBitmap.height

            // For each lane
            for (laneIdx in laneCoefficients.indices) {
                val coeffs = laneCoefficients[laneIdx]

                // Skip lanes with all zero coefficients (likely non-existent lanes)
                if (coeffs.all { Math.abs(it) < 0.01f }) {
                    continue
                }

                // Create paint for this lane
                val paint = Paint().apply {
                    color = laneColors[laneIdx % laneColors.size]
                    strokeWidth = 5f
                    style = Paint.Style.STROKE
                }

                // Calculate points along the polynomial curve
                val points = mutableListOf<Pair<Float, Float>>()
                val numPoints = 100 // Number of points to sample along the curve

                // Generate x values from 0 to 1 (normalized)
                for (i in 0 until numPoints) {
                    val x = i.toFloat() / (numPoints - 1)

                    // Evaluate polynomial
                    var y = 0f
                    for (degree in coeffs.indices) {
                        y += coeffs[degree] * x.pow(degree)
                    }

                    // Convert normalized coordinates to image coordinates
                    // Note: Need to adjust y based on your model's output format
                    // Typically y is normalized from top (0) to bottom (1)
                    val imgX = x * bmpWidth
                    val imgY = y * bmpHeight

                    // Only add points that are within the image boundaries
                    if (imgY >= 0 && imgY < bmpHeight) {
                        points.add(Pair(imgX, imgY))
                    }
                }

                // Draw the points and connect them with lines
                for (i in 0 until points.size - 1) {
                    val (x1, y1) = points[i]
                    val (x2, y2) = points[i + 1]
                    canvas.drawLine(x1, y1, x2, y2, paint)
                }

                // Optional: Draw points at each vertex
                val pointPaint = Paint().apply {
                    color = Color.WHITE
                    style = Paint.Style.FILL
                }
                for ((x, y) in points) {
                    canvas.drawCircle(x, y, 3f, pointPaint)
                }
            }

            // Add debug information
            val textPaint = Paint().apply {
                color = Color.WHITE
                textSize = 30f
                style = Paint.Style.FILL
                setShadowLayer(3f, 1f, 1f, Color.BLACK)
            }
            canvas.drawText("Inference: $inferenceTime ms", 20f, 40f, textPaint)

        } catch (e: Exception) {
            Log.e(TAG, "Error drawing lanes: ${e.message}")
            e.printStackTrace()
            // Add error text to the bitmap
            val textPaint = Paint().apply {
                color = Color.RED
                textSize = 40f
            }
            canvas.drawText("Error visualizing lanes: ${e.message}", 20f, 100f, textPaint)
        }

        return resultBitmap
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}