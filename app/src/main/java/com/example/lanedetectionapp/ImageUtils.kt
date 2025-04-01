package com.example.lanedetectionapp

import android.annotation.SuppressLint
import android.graphics.*
import android.media.Image
import android.util.Log
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

object ImageUtils {
    private const val TAG = "ImageUtils"

    @SuppressLint("UnsafeOptInUsageError")
    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val image = imageProxy.image ?: return null

        try {
            // Convert ImageProxy to NV21 byte array
            val nv21 = yuv420ToNV21(image)

            // Convert NV21 to Bitmap
            val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)

            val imageBytes = out.toByteArray()
            var bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

            // Apply rotation correction
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            if (rotationDegrees != 0) {
                val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
                // Create a new rotated bitmap
                val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

                // Only recycle the original if a new one was created
                if (rotatedBitmap != bitmap) {
                    bitmap.recycle()
                }
                bitmap = rotatedBitmap
            }

            // Clean up
            out.close()
            return bitmap
        } catch (e: Exception) {
            Log.e(TAG, "Error converting image: ${e.message}")
            e.printStackTrace()
            return null
        }
        // Note: We don't close the imageProxy here, as it should be closed by the caller
    }

    private fun yuv420ToNV21(image: Image): ByteArray {
        val width = image.width
        val height = image.height

        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        // NV21 format: YYYYYYYY...VUVUVU...
        val nv21 = ByteArray(width * height * 3 / 2) // YUV420 size

        // Copy Y
        yBuffer.get(nv21, 0, ySize)

        // Copy VU (interleaved as NV21 requires)
        // YUV420 has a specific layout with padding
        // We need to carefully copy U and V values
        val uvWidth = width / 2
        val uvHeight = height / 2
        val uvPixelStride = image.planes[1].pixelStride
        val uvRowStride = image.planes[1].rowStride

        // UV planes may have stride padding
        var uvPos = 0
        val ySize_aligned = width * height

        for (row in 0 until uvHeight) {
            // Calculate position in U and V buffers
            val uvBufferPos = row * uvRowStride

            for (col in 0 until uvWidth) {
                // Calculate NV21 position (V first, then U)
                val pos = ySize_aligned + uvPos * 2

                // Here we swap U and V to get NV21 format
                nv21[pos] = vBuffer.get(uvBufferPos + col * uvPixelStride)      // V
                nv21[pos + 1] = uBuffer.get(uvBufferPos + col * uvPixelStride)  // U

                uvPos++
            }
        }

        return nv21
    }

    // Alternative implementation using direct copying
    private fun yuv420ToNV21_simpler(image: Image): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // Copy Y plane as is
        yBuffer.get(nv21, 0, ySize)

        // Copy V plane first, then U (NV21 format has VU order after Y)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        return nv21
    }

    // Helper to convert YUV to RGB directly if needed
    @Suppress("unused")
    private fun yuv420ToRgb(image: Image): Bitmap {
        val width = image.width
        val height = image.height

        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        // Get the YUV data
        val ySize = yBuffer.remaining()
        val yData = ByteArray(ySize)
        yBuffer.get(yData)

        val uSize = uBuffer.remaining()
        val uData = ByteArray(uSize)
        uBuffer.get(uData)

        val vSize = vBuffer.remaining()
        val vData = ByteArray(vSize)
        vBuffer.get(vData)

        // Allocate RGB buffer
        val argbData = IntArray(width * height)

        // Convert YUV to RGB - simplified conversion algorithm
        val uvRowStride = image.planes[1].rowStride
        val uvPixelStride = image.planes[1].pixelStride

        // Do the conversion row by row
        for (y in 0 until height) {
            val yRowOffset = y * width
            val uvRowOffset = (y shr 1) * uvRowStride

            for (x in 0 until width) {
                // Y component
                val yIndex = yRowOffset + x
                val yValue = yData[yIndex].toInt() and 0xFF

                // UV indices
                val uvIndex = uvRowOffset + (x shr 1) * uvPixelStride
                var uValue = uData[uvIndex].toInt() and 0xFF
                var vValue = vData[uvIndex].toInt() and 0xFF

                // Adjust for YUV values
                uValue -= 128
                vValue -= 128

                // YUV to RGB conversion
                val y1192 = 1192 * (yValue - 16)
                var r = (y1192 + 1634 * vValue) shr 10
                var g = (y1192 - 833 * vValue - 400 * uValue) shr 10
                var b = (y1192 + 2066 * uValue) shr 10

                // Clamp values
                r = r.coerceIn(0, 255)
                g = g.coerceIn(0, 255)
                b = b.coerceIn(0, 255)

                // Set ARGB value
                argbData[yIndex] = -0x1000000 or (r shl 16) or (g shl 8) or b
            }
        }

        // Create bitmap from ARGB data
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(argbData, 0, width, 0, 0, width, height)
        return bitmap
    }
}