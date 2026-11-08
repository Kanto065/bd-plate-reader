package com.bdplatereader

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.util.Base64
import com.facebook.react.ReactPackage
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.uimanager.ViewManager
import java.io.IOException
import kotlin.math.max
import kotlin.math.min

/**
 * Decodes a photo with the platform JPEG decoder, applies the EXIF rotation and
 * shrinks it so the long side is at most maxSide. Returns upright RGB bytes as
 * base64. This replaces the slow JavaScript decode.
 */
class PlateImageModule(context: ReactApplicationContext) : ReactContextBaseJavaModule(context) {
  override fun getName() = "PlateImage"

  @ReactMethod
  fun decode(uri: String, maxSide: Int, promise: Promise) {
    Thread {
      try {
        promise.resolve(decodeUpright(Uri.parse(uri), maxSide))
      } catch (e: Throwable) {
        promise.reject("E_DECODE", e.message, e)
      }
    }.start()
  }

  private fun open(uri: Uri) =
    reactApplicationContext.contentResolver.openInputStream(uri)
      ?: throw IOException("Cannot open $uri")

  private fun decodeUpright(uri: Uri, maxSide: Int) = run {
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
    val out = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)

    val w = out.width
    val h = out.height
    val px = IntArray(w * h)
    out.getPixels(px, 0, w, 0, 0, w, h)
    val rgb = ByteArray(w * h * 3)
    for (i in px.indices) {
      val p = px[i]
      rgb[i * 3] = (p shr 16).toByte()
      rgb[i * 3 + 1] = (p shr 8).toByte()
      rgb[i * 3 + 2] = p.toByte()
    }
    Arguments.createMap().apply {
      putInt("width", w)
      putInt("height", h)
      putString("data", Base64.encodeToString(rgb, Base64.NO_WRAP))
    }
  }
}

class PlateImagePackage : ReactPackage {
  override fun createNativeModules(context: ReactApplicationContext): List<NativeModule> =
    listOf(PlateImageModule(context))

  override fun createViewManagers(context: ReactApplicationContext): List<ViewManager<*, *>> =
    emptyList()
}
