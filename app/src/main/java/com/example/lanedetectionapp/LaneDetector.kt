package com.example.lanedetectionapp

import android.content.Context
import android.graphics.*
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.math.*

class LaneDetector(private val context: Context) {
    private val TAG = "LaneDetector"
    private var interpreter: Interpreter? = null

    // Define model parameters based on the UltraFast Lane Detection model
    private val modelName = "model_float32.tflite" // Your TFLite model name

    // Changed resolution
    private val inputImageWidth = 400  // Changed from 800
    private val inputImageHeight = 144  // Changed from 288

    // Original model resolution - we'll use this for proper scaling
    private val originalModelWidth = 800
    private val originalModelHeight = 288

    private val numChannels = 3  // RGB

    // These values are from the TuSimple dataset as used in the original model
    private val tusimpleRowAnchors = intArrayOf(64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
        116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
        168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
        220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
        272, 276, 280, 284)

    private val gridingNum = 100
    private val numRowAnchors = tusimpleRowAnchors.size
    private val numLanes = 4  // Maximum number of lanes to detect

    // Lane filtering parameters
    private val laneConfidenceThreshold = 0.7f  // Confidence threshold for lane detection
    private val expectedLanesCount = 2  // We expect to see only two lanes

    // Lane validation parameters - updated for new resolution
    private val minLaneWidth = 0.15f * inputImageWidth  // Minimum lane width in pixels
    private val maxLaneWidth = 0.5f * inputImageWidth   // Maximum lane width in pixels

    // Temporal smoothing parameters
    private val maxHistoryFrames = 5
    private val laneHistory = Array(numLanes) { ArrayDeque<MutableList<Pair<Float, Float>>>() }

    // Lane colors for visualization
    private val laneColors = arrayOf(
        Color.GREEN, Color.BLUE, Color.RED, Color.YELLOW
    )

    // Performance tracking
    private var inferenceTime: Long = 0

    init {
        try {
            // Load the TFLite model
            val tfliteModel = loadModelFile()
            val options = Interpreter.Options()
            // Enable hardware acceleration if available
            options.setNumThreads(4) // Adjust based on your device
            // Disable NNAPI as it might not support all ops
            options.setUseNNAPI(false)
            interpreter = Interpreter(tfliteModel, options)
            Log.d(TAG, "TFLite model loaded successfully")

            // Log model info for debugging
            if (interpreter != null) {
                val inputTensor = interpreter!!.getInputTensor(0)
                val outputTensor = interpreter!!.getOutputTensor(0)
                Log.d(TAG, "Model input shape: ${inputTensor.shape().contentToString()}")
                Log.d(TAG, "Model output shape: ${outputTensor.shape().contentToString()}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading TFLite model: ${e.message}")
            e.printStackTrace()
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
        // Apply ROI mask first to focus only on road area
        val maskedBitmap = applyROIMask(bitmap)

        // First resize to the original model's expected input resolution
        // This ensures the lane positions are consistent with what the model expects
        val originalSizeBitmap = Bitmap.createScaledBitmap(
            maskedBitmap, originalModelWidth, originalModelHeight, true)

        // Allocate ByteBuffer for model input based on original model dimensions
        // Since inference will always happen at the original dimensions
        val modelInputSize = originalModelWidth * originalModelHeight * numChannels * 4 // 4 bytes per float
        val inputBuffer = ByteBuffer.allocateDirect(modelInputSize)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Normalization values from the original model
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        // Convert bitmap to normalized float values - this is for the TuSimple model
        // Process the originalSizeBitmap at the model's expected dimensions
        for (y in 0 until originalModelHeight) {
            for (x in 0 until originalModelWidth) {
                val pixel = originalSizeBitmap.getPixel(x, y)
                // Extract RGB values (0-255)
                val r = Color.red(pixel) / 255.0f
                val g = Color.green(pixel) / 255.0f
                val b = Color.blue(pixel) / 255.0f

                // Apply normalization (same as original model)
                inputBuffer.putFloat((r - mean[0]) / std[0])
                inputBuffer.putFloat((g - mean[1]) / std[1])
                inputBuffer.putFloat((b - mean[2]) / std[2])
            }
        }

        // Clean up intermediate bitmaps
        if (maskedBitmap != bitmap) {
            maskedBitmap.recycle()
        }
        originalSizeBitmap.recycle()

        // Reset buffer position
        inputBuffer.rewind()
        return inputBuffer
    }

    private fun applyROIMask(bitmap: Bitmap): Bitmap {
        val result = bitmap.copy(bitmap.config, true)
        val canvas = Canvas(result)
        val paint = Paint().apply {
            color = Color.BLACK
            style = Paint.Style.FILL
            alpha = 180  // Semi-transparent
        }

        val width = bitmap.width
        val height = bitmap.height

        // Define a path for ROI (focus on lower 2/3 of the image)
        val roiPath = Path()

        // Create a mask for the road area (trapezoidal shape)
        roiPath.moveTo(width * 0.1f, height * 0.6f)  // Top-left
        roiPath.lineTo(width * 0.9f, height * 0.6f)  // Top-right
        roiPath.lineTo(width.toFloat(), height.toFloat())  // Bottom-right
        roiPath.lineTo(0f, height.toFloat())  // Bottom-left
        roiPath.close()

        // Create clipPath for everything outside ROI
        val clipPath = Path()
        clipPath.addRect(0f, 0f, width.toFloat(), height.toFloat(), Path.Direction.CW)
        clipPath.op(roiPath, Path.Op.DIFFERENCE)

        canvas.drawPath(clipPath, paint)

        return result
    }

    fun detectLanesPoints(inputBitmap: Bitmap): Pair<Array<List<Pair<Float, Float>>>, Long> {
        if (interpreter == null) {
            Log.e(TAG, "Interpreter is not initialized")
            return Pair(Array(expectedLanesCount) { emptyList<Pair<Float, Float>>() }, 0L)
        }

        val startTime = SystemClock.uptimeMillis()

        try {
            // Step 1: Apply preprocessing with ROI mask - this will handle the resolution conversion
            val inputBuffer = preprocess(inputBitmap)

            // Step 2: Run inference on the model at original resolution
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            val outputBuffer = Array(1) {
                Array(outputShape[1]) {
                    Array(outputShape[2]) {
                        FloatArray(outputShape[3])
                    }
                }
            }
            interpreter!!.run(inputBuffer, outputBuffer)

            // Step 3: Process the output to get lane points - scaled to our target resolution
            val lanePoints = processOutput(outputBuffer[0])

            inferenceTime = SystemClock.uptimeMillis() - startTime
            Log.d(TAG, "Total inference completed in $inferenceTime ms")

            return Pair(lanePoints, inferenceTime)
        } catch (e: Exception) {
            Log.e(TAG, "Error during lane detection: ${e.message}")
            e.printStackTrace()
            inferenceTime = SystemClock.uptimeMillis() - startTime

            // Return empty lane points
            return Pair(Array(expectedLanesCount) { emptyList<Pair<Float, Float>>() }, inferenceTime)
        }
    }

    private fun processOutput(output: Array<Array<FloatArray>>): Array<List<Pair<Float, Float>>> {
        // Detect all possible lanes first - with scaling to our target resolution
        val allDetectedLanes = detectLanesFromOutput(output)

        // Filter lanes based on quality metrics
        val filteredLanes = filterLanes(allDetectedLanes)

        // Update lane history for temporal smoothing
        for (laneIdx in 0 until filteredLanes.size) {
            if (laneHistory[laneIdx].size >= maxHistoryFrames) {
                laneHistory[laneIdx].removeFirst()
            }
            laneHistory[laneIdx].addLast(filteredLanes[laneIdx])
        }

        // Apply temporal smoothing
        val smoothedLanes = smoothLanes()

        // Validate lane positions based on expected geometry
        return validateLanePositions(smoothedLanes)
    }

    private fun detectLanesFromOutput(output: Array<Array<FloatArray>>): Array<MutableList<Pair<Float, Float>>> {
        val lanes = Array(numLanes) { mutableListOf<Pair<Float, Float>>() }

        // Scale factor for resolution change (original model expects 800×288)
        val scaleX = inputImageWidth.toFloat() / originalModelWidth
        val scaleY = inputImageHeight.toFloat() / originalModelHeight

        // Process each of the 4 possible lanes
        for (laneIdx in 0 until numLanes) {
            // Process softmax across the grid for each row
            for (rowIdx in 0 until numRowAnchors) {
                // Extract softmax probabilities
                val probs = FloatArray(gridingNum + 1)
                for (i in 0 until min(gridingNum + 1, output.size)) {
                    probs[i] = output[i][rowIdx][laneIdx]
                }

                // Apply softmax
                val softmax = applySoftmax(probs)

                // Find max probability and index
                var maxProb = 0f
                var maxIdx = -1
                for (i in softmax.indices) {
                    if (softmax[i] > maxProb) {
                        maxProb = softmax[i]
                        maxIdx = i
                    }
                }

                // Only add points with high confidence and not the "no lane" class
                if (maxProb > 0.2f && maxIdx < gridingNum) {
                    // Calculate x coordinate in original resolution space
                    val originalX = maxIdx.toFloat() * originalModelWidth / gridingNum

                    // Get y coordinate from row anchor in original resolution space
                    val originalY = tusimpleRowAnchors[rowIdx].toFloat()

                    // Scale to our target resolution
                    val x = originalX * scaleX
                    val y = originalY * scaleY

                    // Add to lane points
                    lanes[laneIdx].add(Pair(x, y))
                }
            }
        }

        return lanes
    }

    private fun applySoftmax(input: FloatArray): FloatArray {
        val output = FloatArray(input.size)
        var sum = 0f

        // Find max for numerical stability
        val max = input.maxOrNull() ?: 0f

        for (i in input.indices) {
            output[i] = exp((input[i] - max).toDouble()).toFloat()
            sum += output[i]
        }

        // Normalize
        if (sum > 0) {
            for (i in output.indices) {
                output[i] /= sum
            }
        }

        return output
    }

    private fun filterLanes(lanes: Array<MutableList<Pair<Float, Float>>>): Array<MutableList<Pair<Float, Float>>> {
        // Score each lane based on quality metrics
        val laneScores = FloatArray(lanes.size)
        for (i in lanes.indices) {
            val lane = lanes[i]

            // Skip lanes with too few points
            if (lane.size < 5) {
                laneScores[i] = 0f
                continue
            }

            // Score based on number of points (more points = more confident)
            val pointScore = min(1.0f, lane.size / 15.0f)

            // Score based on lane straightness (lower variance = straighter lane)
            val xCoords = lane.map { it.first }
            val yCoords = lane.map { it.second }
            val xVariance = calculateVariance(xCoords)

            // Adjust variance threshold based on new resolution
            val varianceThreshold = 5000f * (inputImageWidth / originalModelWidth.toFloat()).pow(2)
            val straightnessScore = 1.0f - min(1.0f, xVariance / varianceThreshold)

            // Calculate final score
            laneScores[i] = (pointScore * 0.4f + straightnessScore * 0.6f)
        }

        // Create a sorted list of lane indices based on scores
        val sortedLaneIndices = laneScores.indices.sortedByDescending { laneScores[it] }

        // Create the filtered result
        val result = Array(numLanes) { mutableListOf<Pair<Float, Float>>() }

        // Keep only top lanes that meet threshold
        var validLanesCount = 0
        for (idx in sortedLaneIndices) {
            if (laneScores[idx] >= laneConfidenceThreshold && validLanesCount < expectedLanesCount) {
                result[validLanesCount] = lanes[idx]
                validLanesCount++
            }
        }

        return result
    }

    private fun calculateVariance(values: List<Float>): Float {
        if (values.isEmpty()) return 0f

        val mean = values.average().toFloat()
        val variance = values.map { (it - mean) * (it - mean) }.average().toFloat()
        return variance
    }

    private fun smoothLanes(): Array<MutableList<Pair<Float, Float>>> {
        val result = Array(numLanes) { mutableListOf<Pair<Float, Float>>() }

        // Process each lane separately
        for (laneIdx in 0 until numLanes) {
            // Skip if no history
            if (laneHistory[laneIdx].isEmpty()) continue

            // Get all unique y-coordinates
            val allYCoords = mutableSetOf<Float>()
            for (frame in laneHistory[laneIdx]) {
                for (point in frame) {
                    allYCoords.add(point.second)
                }
            }

            // For each y-coordinate, find average x position
            for (y in allYCoords.sorted()) {
                var totalX = 0f
                var count = 0

                // Find points for this y-coordinate across all historical frames
                for (frame in laneHistory[laneIdx]) {
                    val matchingPoints = frame.filter { abs(it.second - y) < 2 } // Allow small tolerance
                    for (point in matchingPoints) {
                        totalX += point.first
                        count++
                    }
                }

                // Add smoothed point if we found matches
                if (count > 0) {
                    result[laneIdx].add(Pair(totalX / count, y))
                }
            }

            // Additional step: apply Bezier smoothing for even smoother curves
            if (result[laneIdx].size > 3) {
                result[laneIdx] = applyBezierSmoothing(result[laneIdx])
            }
        }

        return result
    }

    private fun applyBezierSmoothing(points: MutableList<Pair<Float, Float>>): MutableList<Pair<Float, Float>> {
        if (points.size <= 3) return points

        // Sort by y-coordinate
        val sortedPoints = points.sortedBy { it.second }

        // Result list with more points for smoother appearance
        val result = mutableListOf<Pair<Float, Float>>()

        // Generate bezier curve points
        for (i in 0 until sortedPoints.size - 1) {
            val p0 = sortedPoints[i]
            val p1 = sortedPoints[i + 1]

            // Add the first point
            result.add(p0)

            // Add intermediate points for smoother curve
            if (i < sortedPoints.size - 2) {
                val p2 = sortedPoints[i + 2]

                // Calculate control point
                val ctrlX = (p0.first + p1.first + p2.first) / 3
                val ctrlY = (p0.second + p1.second + p2.second) / 3

                // Add points along the curve
                for (t in 1..3) {
                    val tt = t / 4f
                    val x = (1-tt)*(1-tt)*p0.first + 2*(1-tt)*tt*ctrlX + tt*tt*p1.first
                    val y = (1-tt)*(1-tt)*p0.second + 2*(1-tt)*tt*ctrlY + tt*tt*p1.second
                    result.add(Pair(x, y))
                }
            }
        }

        // Add last point
        result.add(sortedPoints.last())

        return result
    }

    private fun validateLanePositions(lanes: Array<MutableList<Pair<Float, Float>>>): Array<List<Pair<Float, Float>>> {
        // If we have less than 2 lanes with points, no need for positioning validation
        if (lanes.count { it.isNotEmpty() } < 2) {
            // Convert to List type for immutability
            return Array(lanes.size) { i -> lanes[i].toList() }
        }

        val result = Array<MutableList<Pair<Float, Float>>>(lanes.size) { mutableListOf() }

        // First, find lanes with sufficient points
        val validLanes = lanes.mapIndexed { index, lane ->
            Pair(index, lane)
        }.filter {
            it.second.size >= 5
        }.sortedBy {
            // Sort by average x position to find leftmost, middle, rightmost lanes
            it.second.map { point -> point.first }.average()
        }

        // We need at least 2 lanes for position validation
        if (validLanes.size < 2) {
            // Copy valid lanes to result
            validLanes.forEach { (idx, lane) ->
                result[idx] = lane
            }
            // Convert to List type for immutability
            return Array(result.size) { i -> result[i].toList() }
        }

        // Find reference y-coordinate for width checking (lower middle of screen)
        val refY = inputImageHeight * 0.75f

        // Calculate lane widths at reference Y
        val laneWidths = mutableListOf<Float>()
        for (i in 0 until validLanes.size - 1) {
            val leftLane = validLanes[i].second
            val rightLane = validLanes[i + 1].second

            // Find x-coordinates at reference Y
            val leftX = interpolateXAtY(leftLane, refY)
            val rightX = interpolateXAtY(rightLane, refY)

            if (leftX != null && rightX != null) {
                val width = rightX - leftX
                laneWidths.add(width)
            }
        }

        // Check if lane widths are reasonable
        val validPairs = mutableListOf<Pair<Int, Int>>() // Pairs of valid adjacent lanes
        for (i in 0 until laneWidths.size) {
            val width = laneWidths[i]
            if (width in minLaneWidth..maxLaneWidth) {
                validPairs.add(Pair(validLanes[i].first, validLanes[i + 1].first))
            }
        }

        // Prioritize lane pairs based on best fit to expected lane width
        val idealLaneWidth = (minLaneWidth + maxLaneWidth) / 2
        val bestPair = validPairs.minByOrNull { pair ->
            val idx = validPairs.indexOf(pair)
            abs(laneWidths[idx] - idealLaneWidth)
        }

        // If we found a valid lane pair, use it
        if (bestPair != null) {
            result[0] = lanes[bestPair.first]
            result[1] = lanes[bestPair.second]
        } else {
            // Fallback: use the two most convincing lanes
            val bestLanes = validLanes.take(min(2, validLanes.size))
            for ((i, pair) in bestLanes.withIndex()) {
                result[i] = lanes[pair.first]
            }
        }

        // Convert to List type for immutability
        return Array(result.size) { i -> result[i].toList() }
    }

    // Helper to interpolate X coordinate at a specific Y value
    private fun interpolateXAtY(points: List<Pair<Float, Float>>, targetY: Float): Float? {
        // Find points above and below targetY
        val sorted = points.sortedBy { it.second }

        // Return null if can't interpolate
        if (sorted.isEmpty() || targetY < sorted.first().second || targetY > sorted.last().second) {
            return null
        }

        // Find surrounding points
        var below = sorted.first()
        var above = sorted.last()

        for (i in 0 until sorted.size - 1) {
            if (sorted[i].second <= targetY && sorted[i + 1].second >= targetY) {
                below = sorted[i]
                above = sorted[i + 1]
                break
            }
        }

        // Handle exact match
        if (below.second == targetY) return below.first
        if (above.second == targetY) return above.first

        // Linear interpolation
        val ratio = (targetY - below.second) / (above.second - below.second)
        return below.first + ratio * (above.first - below.first)
    }

    // Getter methods for model input dimensions
    fun getInputWidth(): Int = inputImageWidth
    fun getInputHeight(): Int = inputImageHeight

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}