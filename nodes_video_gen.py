"""Video generation node (submit + poll pattern)."""
import logging
import os
import time
import uuid

from .utils import download_bytes, http_request, run_async, tensor_to_base64

logger = logging.getLogger(__name__)


def _resolve_field(data: dict, dot_path: str):
    """Resolve a dot-separated path like 'data.task_id' from a dict."""
    parts = dot_path.strip().split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


class BYOKey_VideoGen:
    """Generic video generation via submit+poll pattern.
    Works with Kling, MiniMax, Runway, Luma, etc.
    Output: path to saved video file (STRING), compatible with video preview nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "submit_url": ("STRING", {"default": "", "tooltip": "POST URL to submit generation task"}),
                "poll_url_template": ("STRING", {"default": "", "tooltip": "GET URL with {task_id} placeholder"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "auth_header_name": (["Authorization", "X-API-Key", "Api-Key"], {"default": "Authorization"}),
                "auth_prefix": ("STRING", {"default": "Bearer "}),
                "task_id_field": ("STRING", {"default": "data.task_id"}),
                "status_field": ("STRING", {"default": "data.task_status"}),
                "video_url_field": ("STRING", {"default": "data.task_result.videos.0.url"}),
                "done_statuses": ("STRING", {"default": "succeed,completed,done"}),
                "fail_statuses": ("STRING", {"default": "failed,error"}),
                "poll_interval": ("INT", {"default": 10, "min": 2, "max": 120}),
                "max_wait": ("INT", {"default": 600, "min": 30, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("Path to generated video file",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey"
    DESCRIPTION = "Video generation via submit+poll. Configure URL templates and field paths for your provider."

    def execute(self, api_key, submit_url, poll_url_template, prompt,
                model="", image=None, auth_header_name="Authorization", auth_prefix="Bearer ",
                task_id_field="data.task_id", status_field="data.task_status",
                video_url_field="data.task_result.videos.0.url",
                done_statuses="succeed,completed,done", fail_statuses="failed,error",
                poll_interval=10, max_wait=600):

        headers = {auth_header_name: f"{auth_prefix}{api_key}", "Content-Type": "application/json"}
        body = {"prompt": prompt}
        if model:
            body["model"] = model
        if image is not None:
            body["image"] = f"data:image/png;base64,{tensor_to_base64(image)}"

        submit_resp = run_async(http_request("POST", submit_url, headers, body))
        task_id = _resolve_field(submit_resp, task_id_field)
        if not task_id:
            raise RuntimeError(f"Could not extract task_id from response using path '{task_id_field}': {submit_resp}")

        done_set = {s.strip().lower() for s in done_statuses.split(",")}
        fail_set = {s.strip().lower() for s in fail_statuses.split(",")}
        poll_url = poll_url_template.replace("{task_id}", str(task_id))

        elapsed = 0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            poll_resp = run_async(http_request("GET", poll_url, headers))
            status = str(_resolve_field(poll_resp, status_field) or "").lower()
            if status in fail_set:
                raise RuntimeError(f"Video generation failed. Status: {status}. Response: {poll_resp}")
            if status in done_set:
                video_url = _resolve_field(poll_resp, video_url_field)
                if not video_url:
                    raise RuntimeError(f"Task done but no video URL at '{video_url_field}': {poll_resp}")
                return (self._download_video(video_url, headers),)

        raise RuntimeError(f"Timed out after {max_wait}s waiting for video generation")

    def _download_video(self, url: str, headers: dict) -> str:
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        filename = f"byokey_video_{uuid.uuid4().hex[:8]}.mp4"
        filepath = os.path.join(output_dir, filename)
        video_bytes = run_async(download_bytes(url))
        with open(filepath, "wb") as f:
            f.write(video_bytes)
        return filepath
