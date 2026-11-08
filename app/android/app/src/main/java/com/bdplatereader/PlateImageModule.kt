package com.bdplatereader

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.media.ExifInterface
import android.net.Uri
import com.facebook.react.ReactPackage
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.WritableMap
import com.facebook.react.uimanager.ViewManager
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Photo handling for the plate reader, done with the platform decoder instead of
 * JavaScript. letterbox() decodes the photo upright (EXIF applied), optionally
 * crops a box, letterboxes it to a size x size square with gray padding, and
 * writes it as the model input: planar RGB (CHW) float32 in 0..1, in a cache
 * file. JavaScript reads the file as a raw buffer, with no per pixel work.
 */
class PlateImageModule(context: ReactApplicationContext) : ReactContextBaseJavaModule(context) {
  override fun getName() = "PlateImage"

  /** crop is [x1, y1, x2, y2] in pixels of the upright decoded photo, or null. */
  @ReactMethod
  fun letterbox(uri: String, maxSide: Int, size: Int, crop: ReadableArray?, promise: Promise) {
    Thread {
      try {
        promise.resolve(letterboxed(Uri.parse(uri), maxSide, size, crop))
      } catch (e: Throwable) {
        promise.reject("E_DECODE", e.message, e)
      }
    }.start()
  }

  // The plate stage and the character stage both need the same decoded photo, so
  // keep the last one.
  // shortcut: holds one decoded photo (about 13 MB) until the next photo, clear it on low memory if that becomes a problem.
  private var cache: Triple<String, Int, Bitmap>? = null

  @Synchronized
  private fun upright(uri: Uri, maxSide: Int): Bitmap {
    cache?.let { (u, m, b) -> if (u == uri.toString() && m == maxSide) return b }
    return loadUpright(uri, maxSide).also { cache = Triple(uri.toString(), maxSide, it) }
  }

  private fun open(uri: Uri) =
    reactApplicationContext.contentResolver.openInputStream(uri)
      ?: throw IOException("Cannot open $uri")

  private fun loadUpright(uri: Uri, maxSide: Int): Bitmap {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    open(uri).use { BitmapFactory.decodeStream(it, null, bounds) }
    if (bounds.outWidth <= 0) throw IOException("Not an image")

    // Power of two subsampling skips pixels inside the decoder, so a 12 MP photo
    // is never fully decoded. Stop while the result is still >= maxSide.
    var sample = 1
    while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2
    val raw = open(uri).use {
      BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = sample })
    } ?: throw IOException("Decode failed")

    val orientation = open(uri).use {
      ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
    }
    val m = Matrix()
    when (orientation) {
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
      ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
      ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
      ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
      ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(-90f); m.postScale(-1f, 1f) }
      ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
    }
    val s = min(1f, maxSide.toFloat() / max(raw.width, raw.height))
    m.postScale(s, s)
    return Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
  }

  private fun letterboxed(uri: Uri, maxSide: Int, size: Int, crop: ReadableArray?): WritableMap {
    var bmp = upright(uri, maxSide)
    val srcWidth = bmp.width
    val srcHeight = bmp.height
    if (crop != null) {
      // Same rounding and clamping as the JavaScript cropRgb.
      val x1 = max(0, min(bmp.width - 1, floor(crop.getDouble(0)).toInt()))
      val y1 = max(0, min(bmp.height - 1, floor(crop.getDouble(1)).toInt()))
      val x2 = max(x1 + 1, min(bmp.width, ceil(crop.getDouble(2)).toInt()))
      val y2 = max(y1 + 1, min(bmp.height, ceil(crop.getDouble(3)).toInt()))
      bmp = Bitmap.createBitmap(bmp, x1, y1, x2 - x1, y2 - y1)
    }

    val scale = min(size.toFloat() / bmp.width, size.toFloat() / bmp.height)
    val newW = max(1, (bmp.width * scale).roundToInt())
    val newH = max(1, (bmp.height * scale).roundToInt())
    val padX = (size - newW) / 2
    val padY = (size - newH) / 2
    val square = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(square)
    canvas.drawColor(0xFF727272.toInt()) // Ultralytics letterbox gray, 114
    canvas.drawBitmap(bmp, null, Rect(padX, padY, padX + newW, padY + newH), Paint(Paint.FILTER_BITMAP_FLAG))

    val px = IntArray(size * size)
    square.getPixels(px, 0, size, 0, 0, size, size)
    val plane = size * size
    val bytes = ByteBuffer.allocate(plane * 3 * 4).order(ByteOrder.nativeOrder())
    val buf = bytes.asFloatBuffer()
    for (i in 0 until plane) {
      val p = px[i]
      buf.put(i, ((p shr 16) and 0xFF) / 255f)
      buf.put(plane + i, ((p shr 8) and 0xFF) / 255f)
      buf.put(2 * plane + i, (p and 0xFF) / 255f)
    }
    // One file per size, so the plate (640) and the crop (416) do not collide.
    val file = File(reactApplicationContext.cacheDir, "plate_input_$size.f32")
    file.writeBytes(bytes.array())
    return Arguments.createMap().apply {
      putInt("srcWidth", srcWidth)
      putInt("srcHeight", srcHeight)
      putDouble("scale", scale.toDouble())
      putInt("padX", padX)
      putInt("padY", padY)
      putString("path", "file://" + file.absolutePath)
    }
  }
}

class PlateImagePackage : ReactPackage {
  override fun createNativeModules(context: ReactApplicationContext): List<NativeModule> =
    listOf(PlateImageModule(context))

  override fun createViewManagers(context: ReactApplicationContext): List<ViewManager<*, *>> =
    emptyList()
}
