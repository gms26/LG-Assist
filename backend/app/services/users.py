"""Anonymous user identity (contract v1).

The frontend generates a UUID once, stores it, and sends it in the
``X-User-Id`` header on every call. The backend creates the user row on
first sight. A missing/invalid header degrades to a fresh anonymous user
instead of erroring, so the API stays usable from curl and new clients.
"""

import uuid

from fastapi import Depends, Header
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User


def get_current_user(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> User:
    external_id: str | None = None
    if x_user_id:
        try:
            external_id = str(uuid.UUID(x_user_id))
        except ValueError:
            external_id = None

    if external_id is not None:
        user = db.query(User).filter(User.external_id == external_id).first()
        if user is not None:
            return user
        user = User(external_id=external_id)
    else:
        user = User()  # anonymous fallback, no external id

    db.add(user)
    db.commit()
    db.refresh(user)
    return user
