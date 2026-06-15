"""Generic configurable video generation node (Bring Your Own Key).

This node has NO single official ComfyUI equivalent. It is a flexible
submit+poll video generator that lets you point it at any vendor that follows
the common "POST a job, then poll a task endpoint until a video URL appears"
pattern (Kling, MiniMax, Runway, Luma, etc.) by configuring the submit URL,
poll URL template, auth header, and dotted field paths for the task id, status,
and result video URL.

Unlike a STRING-path node, the output is a real ComfyUI ``VIDEO`` object
(built via ``utils.download_video_output``) so it plugs straight into the same
preview/save nodes the official API video nodes use. Polling is delegated to
``utils.poll_until``.
"""

import logging

from .utils import (
    download_video_output,
    http_request,
    poll_until,
    run_async,
    tensor_to_base64,
    validate_prompt,
)

logger = logging.getLogger("comfyui_byokey")

# Auth header schemes the generic node can build. The value is the header name;
# the prefix (e.g. "Bearer ") is supplied separately so relays/vendors that use
# a bare token also work.
AUTH_HEADER_NAMES = ["Authorization", "X-API-Key", "Api-Key", "x-api-key"]


def _resolve_field(data, dot_path: str):
    """Resolve a dotted path like ``data.task_result.videos.0.url`` from JSON.

    Supports integer indices into lists (numeric path segments) as well as dict
    keys, so result paths into arrays of videos resolve correctly.
    """
    if not dot_path:
        return None
    current = data
    for part in dot_path.strip().split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, (list, tuple)):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if current is None:
            return None
    return current


class BYOKey_VideoGen:
    """Generic submit+poll video generation (bring your own key).

    Configure the submit URL, poll URL template (with a ``{task_id}``
    placeholder), the auth header scheme, and the dotted field paths used to
    extract the task id, status, and final video URL from the vendor JSON.
    Outputs a ComfyUI ``VIDEO`` object, exactly like the official API video
    nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Vendor API key / token."}),
                "base_url": ("STRING", {
                    "default": "https://api.example.com",
                    "tooltip": "Vendor API host. Used to resolve relative submit/poll paths. Override for relays/proxies.",
                }),
                "submit_url": ("STRING", {
                    "default": "",
                    "tooltip": "POST URL (or path relative to base_url) to submit the generation task.",
                }),
                "poll_url_template": ("STRING", {
                    "default": "",
                    "tooltip": "GET URL (or path relative to base_url) with a {task_id} placeholder to poll status.",
                }),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text prompt."}),
            },
            "optional": {
                "model": ("STRING", {"default": "", "tooltip": "Optional model id to include in the request body."}),
                "image": ("IMAGE", {"tooltip": "Optional reference image (sent as a base64 data URI)."}),
                "auth_header_name": (AUTH_HEADER_NAMES, {"default": "Authorization", "tooltip": "HTTP header used to send the credential."}),
                "auth_prefix": ("STRING", {"default": "Bearer ", "tooltip": "Prefix prepended to the api_key in the auth header (use empty for a bare token)."}),
                "task_id_field": ("STRING", {"default": "data.task_id", "tooltip": "Dotted path to the task id in the submit response."}),
                "status_field": ("STRING", {"default": "data.task_status", "tooltip": "Dotted path to the status in the poll response."}),
                "video_url_field": ("STRING", {"default": "data.task_result.videos.0.url", "tooltip": "Dotted path to the result video URL in the poll response (numeric segments index lists)."}),
                "done_statuses": ("STRING", {"default": "succeed,completed,done,success", "tooltip": "Comma-separated statuses that mean success."}),
                "fail_statuses": ("STRING", {"default": "failed,error", "tooltip": "Comma-separated statuses that mean failure."}),
                "poll_interval": ("INT", {"default": 10, "min": 1, "max": 120, "tooltip": "Seconds between status polls."}),
                "max_wait": ("INT", {"default": 600, "min": 30, "max": 3600, "tooltip": "Max seconds to wait for completion."}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("output",)
    OUTPUT_TOOLTIPS = ("Generated video.",)
    FUNCTION = "execute"
    CATEGORY = "api/byokey/video"
    DESCRIPTION = "Generic submit+poll video generation. Configure URLs, auth, and field paths for your provider; outputs a VIDEO."

    @staticmethod
    def _resolve_url(base: str, url_or_path: str) -> str:
        """Return an absolute URL: use ``url_or_path`` as-is if absolute, else join to base."""
        url_or_path = (url_or_path or "").strip()
        if url_or_path.startswith(("http://", "https://")):
            return url_or_path
        base = (base or "").rstrip("/")
        if not url_or_path:
            return base
        return f"{base}/{url_or_path.lstrip('/')}"

    def execute(self, api_key, base_url, submit_url, poll_url_template, prompt,
                model="", image=None, auth_header_name="Authorization", auth_prefix="Bearer ",
                task_id_field="data.task_id", status_field="data.task_status",
                video_url_field="data.task_result.videos.0.url",
                done_statuses="succeed,completed,done,success", fail_statuses="failed,error",
                poll_interval=10, max_wait=600):
        validate_prompt(prompt, min_length=1)

        submit_endpoint = self._resolve_url(base_url, submit_url)
        if not submit_endpoint:
            raise ValueError("submit_url (or base_url) must be provided.")
        poll_template = poll_url_template.strip()
        if not poll_template:
            raise ValueError("poll_url_template must be provided.")

        headers = {
            auth_header_name: f"{auth_prefix}{api_key}",
            "Content-Type": "application/json",
        }

        body = {"prompt": prompt}
        if model:
            body["model"] = model
        if image is not None:
            body["image"] = f"data:image/png;base64,{tensor_to_base64(image, 'image/png')}"

        done_set = [s.strip() for s in done_statuses.split(",") if s.strip()]
        fail_set = [s.strip() for s in fail_statuses.split(",") if s.strip()]

        return run_async(
            self._submit_and_poll(
                submit_endpoint, poll_template, body, headers, base_url,
                task_id_field, status_field, video_url_field,
                done_set, fail_set, poll_interval, max_wait,
            )
        )

    async def _submit_and_poll(self, submit_endpoint, poll_template, body, headers, base_url,
                               task_id_field, status_field, video_url_field,
                               done_set, fail_set, poll_interval, max_wait):
        submit_resp = await http_request("POST", submit_endpoint, headers, body)
        task_id = _resolve_field(submit_resp, task_id_field)
        if task_id is None or task_id == "":
            raise RuntimeError(
                f"Could not extract task id from submit response using path "
                f"'{task_id_field}': {submit_resp}"
            )

        poll_path = poll_template.replace("{task_id}", str(task_id))
        poll_url = self._resolve_url(base_url, poll_path)

        async def poll_fn():
            return await http_request("GET", poll_url, headers)

        def status_of(resp):
            return _resolve_field(resp, status_field)

        final = await poll_until(
            poll_fn,
            status_of=status_of,
            done_statuses=done_set,
            failed_statuses=fail_set,
            interval=poll_interval,
            max_wait=max_wait,
            on_fail_message=f"Video generation task {task_id} failed",
        )

        video_url = _resolve_field(final, video_url_field)
        if not video_url:
            raise RuntimeError(
                f"Task {task_id} completed but no video URL was found at "
                f"'{video_url_field}': {final}"
            )

        video_out = await download_video_output(str(video_url), headers=headers)
        return (video_out,)
