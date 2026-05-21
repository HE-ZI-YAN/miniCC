from __future__ import annotations

from pydantic import BaseModel


class GUIResult(BaseModel):
    action: str
    success: bool
    message: str


class GUIAgent:
    """Optional desktop automation adapter.

    If pyautogui is installed, basic screenshot/click/type operations work.
    Vision and omni-parser integration belong behind this interface.
    """

    def __init__(self):
        try:
            import pyautogui  # type: ignore
        except Exception:
            pyautogui = None
        self.pyautogui = pyautogui

    def screenshot(self, path: str = "screenshot.png") -> GUIResult:
        if not self.pyautogui:
            return GUIResult(action="screenshot", success=False, message="pyautogui is not installed")
        image = self.pyautogui.screenshot()
        image.save(path)
        return GUIResult(action="screenshot", success=True, message=path)

    def click(self, x: int, y: int) -> GUIResult:
        if not self.pyautogui:
            return GUIResult(action="click", success=False, message="pyautogui is not installed")
        self.pyautogui.click(x, y)
        return GUIResult(action="click", success=True, message=f"clicked {x},{y}")

    def type_text(self, text: str) -> GUIResult:
        if not self.pyautogui:
            return GUIResult(action="type_text", success=False, message="pyautogui is not installed")
        self.pyautogui.write(text)
        return GUIResult(action="type_text", success=True, message=f"typed {len(text)} characters")
