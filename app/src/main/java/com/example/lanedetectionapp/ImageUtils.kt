//package com.example.lanedetectionapp
//
//import android.graphics.Bitmap
//import android.graphics.BitmapFactory
//import android.graphics.ImageFormat
//import android.graphics.Matrix
//import android.graphics.Rect
//import android.graphics.YuvImage
//import android.media.Image
//import android.util.Log
//import androidx.camera.core.ImageProxy
//import java.io.ByteArrayOutputStream
//
//object ImageUtils {
//    private const val TAG = "ImageUtils"
//
//    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
//        val image = imageProxy.image ?: return null
//
//        try {
//            // This is a more reliable conversion from YUV to RGB for camera preview
//            val planes = image.planes
//            val yBuffer = planes[0].buffer
//            val uBuffer = planes[1].buffer
//            val vBuffer = planes[2].buffer
//
//            val ySize = yBuffer.remaining()
//            val uSize = uBuffer.remaining()
//            val vSize = vBuffer.remaining()
//
//            val nv21 = ByteArray(ySize + uSize + vSize)
//
//            // U and V are swapped
//            yBuffer.get(nv21, 0, ySize)
//            vBuffer.get(nv21, ySize, vSize)
//            uBuffer.get(nv21, ySize + vSize, uSize)
//
//            val yuvImage = YuvImage(
//                nv21,
//                ImageFormat.NV21,
//                image.width,
//                image.height,
//                null
//            )
//
//            val out = ByteArrayOutputStream()
//            yuvImage.compressToJpeg(
//                Rect(0, 0, image.width, image.height),
//                100,
//                out
//            )
//
//            val imageBytes = out.toByteArray()
//            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
//
//            // Apply rotation if needed based on image rotation
//            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
//            if (rotationDegrees != 0) {
//                val matrix = Matrix()
//                matrix.postRotate(rotationDegrees.toFloat())
//                return Bitmap.createBitmap(
//                    bitmap,
//                    0,
//                    0,
//                    bitmap.width,
//                    bitmap.height,
//                    matrix,
//                    true
//                )
//            }
//
//            return bitmap
//        } catch (e: Exception) {
//            Log.e(TAG, "Error converting image: ${e.message}")
//            e.printStackTrace()
//            return null
//        }
//    }
//}

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
                bitmap = rotateBitmap(bitmap, rotationDegrees)
            }

            return bitmap
        } catch (e: Exception) {
            Log.e(TAG, "Error converting image: ${e.message}")
            return null
        } finally {
            imageProxy.close()
        }
    }

    private fun yuv420ToNV21(image: Image): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize) // Swap U and V for NV21 format
        uBuffer.get(nv21, ySize + vSize, uSize)

        return nv21
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}