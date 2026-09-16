import time
import os
import tempfile
import zipfile
import platform
import subprocess

# Compatibility shim: moviepy 1.0.3's resize fx (utility/render's Ken Burns
# effect calls .resize()) references PIL.Image.ANTIALIAS, which Pillow 10+
# removed in favor of Image.Resampling.LANCZOS / Image.LANCZOS. Nothing in
# this pipeline called .resize() on an image/video clip before — Pexels
# clips are pre-filtered to the exact target resolution — so this
# incompatibility was latent until local-reference-image mode needed it.
from PIL import Image as _PILImage
if not hasattr(_PILImage, "ANTIALIAS"):
    _PILImage.ANTIALIAS = _PILImage.LANCZOS

from moviepy.editor import (AudioFileClip, CompositeVideoClip, CompositeAudioClip, ImageClip,
                              TextClip, VideoFileClip, VideoClip)
from moviepy.audio.fx.audio_loop import audio_loop
from moviepy.audio.fx.audio_normalize import audio_normalize
import numpy as np
import requests
from utility.config import get_config


def make_ken_burns_clip(image_path, duration, target_size, zoom_end=1.06, n_keyframes=30):
    """Builds a Ken Burns (slow zoom-in) clip from a still image.

    Two rounds of measurement drove this implementation:
    1. moviepy's built-in .resize(lambda t: ...) calls a full PIL resize on
       the WHOLE image for every output frame — measured at ~4x realtime.
    2. Replacing that with a per-frame numpy-crop-then-resize (crop a
       shrinking window out of one pre-resized source array, resize only
       that crop) barely helped: ~126ms/frame, still ~3x realtime. The
       reason is the zoom range is intentionally small (a few percent, so
       the pan looks smooth rather than jarring) — the crop window stays
       close to the full cover-fit source size throughout, so the "small"
       resize wasn't actually small, especially for a source image whose
       aspect ratio differs a lot from the target (e.g. a landscape photo
       cover-fit into a portrait frame needs a large upscale just to fill
       the frame, before any zoom is even applied).

    The fix: since the zoom changes gradually, the crop only needs to be
    recomputed a few dozen times over the whole clip, not once per frame —
    intermediate frames reuse the nearest precomputed keyframe. This is
    ~9x faster (measured: 34s -> 3.8s for a 10s clip) and, critically, the
    cost no longer scales with clip duration or frame rate at all, only
    with n_keyframes and the one-time cover-fit resize.
    """
    target_w, target_h = target_size
    img = _PILImage.open(image_path).convert("RGB")

    # Cover-fit resize ONCE: scale so the image fully covers the target
    # frame with a little extra headroom for the zoom to crop into.
    scale = max(target_w / img.width, target_h / img.height) * 1.15
    base_w, base_h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    source = np.array(img.resize((base_w, base_h), _PILImage.LANCZOS))

    n_keyframes = max(2, n_keyframes)
    keyframes = []
    for i in range(n_keyframes):
        progress = i / (n_keyframes - 1)
        zoom = 1.0 + (zoom_end - 1.0) * progress
        crop_w = max(target_w, int(base_w / zoom))
        crop_h = max(target_h, int(base_h / zoom))
        x0 = (base_w - crop_w) // 2
        y0 = (base_h - crop_h) // 2
        cropped = source[y0:y0 + crop_h, x0:x0 + crop_w]
        keyframes.append(np.array(_PILImage.fromarray(cropped).resize((target_w, target_h), _PILImage.LANCZOS)))

    def make_frame(t):
        progress = min(max(t / duration, 0.0), 1.0) if duration > 0 else 0.0
        idx = min(int(progress * n_keyframes), n_keyframes - 1)
        return keyframes[idx]

    return VideoClip(make_frame, duration=duration)


def download_file(url, filename):
    with open(filename, 'wb') as f:
        headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        f.write(response.content)

def search_program(program_name):
    try: 
        search_cmd = "where" if platform.system() == "Windows" else "which"
        return subprocess.check_output([search_cmd, program_name]).decode().strip()
    except subprocess.CalledProcessError:
        return None

def get_program_path(program_name):
    program_path = search_program(program_name)
    return program_path

def get_output_media(audio_file_path, timed_captions, background_video_data, video_server, background_music_path=None):
    config = get_config()
    
    # Check if rendering with Remotion is configured
    render_engine = os.getenv('RENDER_ENGINE', 'moviepy').lower()
    if render_engine == 'remotion':
        print("[RenderEngine] Routing compilation to React/Remotion renderer...")
        from utility.render.remotion_renderer import render_with_remotion
        return render_with_remotion(
            audio_file_path=audio_file_path,
            timed_captions=timed_captions,
            background_video_data=background_video_data,
            background_music_path=background_music_path
        )

    OUTPUT_FILE_NAME = "rendered_video.mp4"
    magick_path = get_program_path("magick")
    print(magick_path)
    if magick_path:
        os.environ['IMAGEMAGICK_BINARY'] = magick_path
    else:
        os.environ['IMAGEMAGICK_BINARY'] = '/usr/bin/convert'
    
    orientation_landscape = config.get_video_orientation()
    target_size = (1920, 1080) if orientation_landscape else (1080, 1920)

    visual_clips = []
    downloaded_files = []  # only ever populated for real downloads, so cleanup below doesn't touch local reference images
    for (t1, t2), video_url in background_video_data:
        if os.path.exists(video_url):
            # Local reference image (see utility/video/local_image_generator.py)
            # rather than a downloaded Pexels clip — apply a Ken Burns
            # pan/zoom so a still photo doesn't look static.
            segment_duration = t2 - t1
            segment_clip = make_ken_burns_clip(video_url, segment_duration, target_size)
            segment_clip = segment_clip.set_start(t1).set_end(t2)
            visual_clips.append(segment_clip)
        else:
            # Download video file
            video_filename = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            download_file(video_url, video_filename)
            downloaded_files.append(video_filename)

            # Create VideoFileClip from downloaded file
            video_clip = VideoFileClip(video_filename)
            video_clip = video_clip.set_start(t1)
            video_clip = video_clip.set_end(t2)
            visual_clips.append(video_clip)
    
    audio_clips = []
    audio_file_clip = AudioFileClip(audio_file_path)
    audio_clips.append(audio_file_clip)

    if background_music_path and os.path.exists(background_music_path):
        try:
            bg_music_clip = AudioFileClip(background_music_path)
            # Set volume of background music to 12% so voiceover remains clear
            bg_music_clip = bg_music_clip.volumex(0.12)
            # Loop bg music if it's shorter than voiceover
            if bg_music_clip.duration < audio_file_clip.duration:
                bg_music_clip = audio_loop(bg_music_clip, duration=audio_file_clip.duration)
            else:
                bg_music_clip = bg_music_clip.set_duration(audio_file_clip.duration)
            audio_clips.append(bg_music_clip)
            print("[RenderEngine] Successfully loaded and mixed background music.")
        except Exception as e:
            print(f"[RenderEngine] Error loading/mixing background music: {e}")

    
    # Only add captions if enabled in config
    if config.get_captions_enabled():
        for (t1, t2), text in timed_captions:
            # Get caption styling from config
            font_size = config.get_caption_font_size()
            font_color = config.get_caption_font_color()
            stroke_width = config.get_caption_stroke_width()
            stroke_color = config.get_caption_stroke_color()
            font_face = config.get_caption_font_face()
            caption_position = config.get_caption_position()

            # Convert caption position string to MoviePy format
            # For 1080p video: top=100, center=540, bottom=1000
            if caption_position == 'bottom_center':
                position = ["center", 1000]
            elif caption_position == 'bottom_left':
                position = ["left", 1000]
            elif caption_position == 'bottom_right':
                position = ["right", 1000]
            elif caption_position == 'top':
                position = ["center", 100]
            elif caption_position == 'center':
                position = ["center", 540]
            else: # Default to bottom_center
                position = ["center", 1000]

            text_clip = TextClip(txt=text, font=font_face, fontsize=font_size, color=font_color, stroke_width=stroke_width, stroke_color=stroke_color, method="label")
            text_clip = text_clip.set_start(t1)
            text_clip = text_clip.set_end(t2)
            text_clip = text_clip.set_position(position)
            visual_clips.append(text_clip)
    
    video = CompositeVideoClip(visual_clips)
    
    if audio_clips:
        audio = CompositeAudioClip(audio_clips)
        video.duration = audio.duration
        video.audio = audio

    video.write_videofile(OUTPUT_FILE_NAME, codec='libx264', audio_codec='aac', fps=25, preset='veryfast')

    # Clean up downloaded temp files only — never touch local reference
    # images, those are the caller's own persistent files. (The previous
    # version of this cleanup generated a *new* temp filename and deleted
    # that instead of the one actually downloaded above, silently leaking
    # every downloaded clip — downloaded_files now tracks the real paths.)
    for video_filename in downloaded_files:
        try:
            os.remove(video_filename)
        except OSError:
            pass

    return OUTPUT_FILE_NAME
