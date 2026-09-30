class AppError(Exception):
    """Base application exception."""


class InvalidFileError(AppError):
    """Raised when an uploaded file is not supported."""


class DocumentNotFoundError(AppError):
    """Raised when a document cannot be found."""


class ProcessingDependencyError(AppError):
    """Raised when an optional processing dependency is missing."""
