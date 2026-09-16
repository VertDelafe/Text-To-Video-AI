"""Builds timed visual segments from local reference images instead of
searching Pexels. Used when the caller supplies their own photo(s) — the
video is assembled from those images (with a Ken Burns pan/zoom applied in
render_engine.py) rather than stock footage matched by an LLM-generated
search query.

This also sidesteps the least reliable part of the pipeline: the B-roll
query generator asks a local LLM for an exact JSON schema it doesn't
reliably produce (see docs/repo-evaluation.md) — reference-image mode
skips that call entirely, so it's both more "clean and related" (the
user's own photos) and faster/more deterministic than the stock-search path.
"""


def build_local_image_segments(image_paths, total_duration, seconds_per_image=6.0):
    """Evenly distributes local images across [0, total_duration], cycling
    through them if there are more segments than images. Returns the same
    [[t1, t2], path_or_url] shape generate_video_url() produces, so
    render_engine.py can handle both uniformly — it tells them apart by
    checking whether the string is an existing local file.
    """
    if not image_paths or total_duration <= 0:
        return []

    n_segments = max(1, round(total_duration / seconds_per_image))
    seg_len = total_duration / n_segments

    segments = []
    t = 0.0
    for i in range(n_segments):
        t_next = total_duration if i == n_segments - 1 else t + seg_len
        image_path = image_paths[i % len(image_paths)]
        segments.append([[round(t, 2), round(t_next, 2)], image_path])
        t = t_next
    return segments
