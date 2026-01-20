class CommentError(Exception):
    def __init__(self, comment: str) -> None:
        self.comment = comment
        super().__init__(f"this is the comment: {self.comment}")
