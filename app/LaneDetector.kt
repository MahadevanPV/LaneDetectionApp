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

class LaneDetector(private val context: Context) {
    private val TAG = "LaneDetector"
    private var interpreter: Interpreter? = null

    // Define model parameters (adjust these based on your specific model)
    private val modelName = "lane_detection_model.tflite"
    private val inputImageWidth = 256  // Change according to your model input size
    private val inputImageHeight = 256  // Change according to your model input size
    private val numChannels = 3  // RGB

    // Performance tracking
    private var inferenceTime: Long = 0

    init {
        try {
            // Load the TFLite model
            val tfliteModel = loadModelFile()
            val options = Interpreter.Options()
            // Enable hardware acceleration if available
            options.setNumThreads(4) // Adjust based on your device
            interpreter = Interpreter(tfliteModel, options)
            Log.d(TAG, "TFLite model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading TFLite model: ${e.message}")
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
            return Pair(inputBitmap, 0L)
        }

        val startTime = SystemClock.uptimeMillis()

        // Prepare input data
        val inputBuffer = preprocess(inputBitmap)

        // Prepare output buffer - Shape depends on your model output
        // This example assumes output shape is [1, H, W, C]
        // where C is number of lane classes or points
        val outputShape = interpreter!!.getOutputTensor(0).shape()
        val outputByteBuffer = ByteBuffer.allocateDirect(
            outputShape[0] * outputShape[1] * outputShape[2] * outputShape[3] * 4) // 4 bytes per float
        outputByteBuffer.order(ByteOrder.nativeOrder())

        // Run inference
        interpreter!!.run(inputBuffer, outputByteBuffer)

        inferenceTime = SystemClock.uptimeMillis() - startTime

        // Process output to draw lanes on the bitmap
        val resultBitmap = drawLanesOnBitmap(inputBitmap, outputByteBuffer, outputShape)

        return Pair(resultBitmap, inferenceTime)
    }

    private fun drawLanesOnBitmap(
        originalBitmap: Bitmap,
        outputBuffer: ByteBuffer,
        outputShape: IntArray
    ): Bitmap {
        // Create a mutable copy of the original bitmap
        val resultBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(resultBitmap)
        val paint = Paint().apply {
            color = Color.GREEN
            strokeWidth = 5f
            style = Paint.Style.STROKE
        }

        // Reset the buffer position to read from the beginning
        outputBuffer.rewind()

        // This implementation depends on your model's output format
        // Below is a simplified example that assumes your model outputs lane coordinates
        // You'll need to adjust this based on your specific model output

        // For simplicity, let's assume our output is [1, H, W, 2] where 2 is for x,y coordinates
        // of lane points for demonstration
        val numLanePoints = outputShape[1]

        // Draw detected lanes
        for (i in 0 until numLanePoints) {
            val x = outputBuffer.getFloat() * originalBitmap.width  // Scale to bitmap width
            val y = outputBuffer.getFloat() * originalBitmap.height // Scale to bitmap height

            // Skip points with very low confidence or invalid coordinates
            if (x > 0 && y > 0 && x < originalBitmap.width && y < originalBitmap.height) {
                canvas.drawCircle(x, y, 5f, paint)

                // Connect points with lines if we have more than one point
                if (i > 0) {
                    // Previous point (this would need to be tracked in a real implementation)
                    // This is simplified for demonstration
                    val prevX = (x - 5).coerceAtLeast(0f)
                    val prevY = (y - 5).coerceAtLeast(0f)
                    canvas.drawLine(prevX, prevY, x, y, paint)
                }
            }
        }

        return resultBitmap
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}