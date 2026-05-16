class AppError(Exception):
    def __init__(self, code: str, message: str, details: dict | None = None, status_code: int = 400):
        self.code = code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(AppError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__("not_found", message, details=details, status_code=404)


class ConflictError(AppError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__("conflict", message, details=details, status_code=409)


class UpstreamError(AppError):
    def __init__(
        self,
        message: str,
        details: dict | None = None,
        *,
        status_code: int = 502,
        code: str = "upstream_error",
    ):
        super().__init__(code, message, details=details, status_code=status_code)
