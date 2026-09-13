from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Deque, Dict, Optional
from zoneinfo import ZoneInfo


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


class AdmissionError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class AdmissionReservation:
    client_ip: str
    timestamp: float
    local_date: date


class AdmissionController:
    """In-memory rolling hourly rate limit plus Shanghai-calendar daily quota."""

    def __init__(
        self,
        rate_limit_per_hour: int,
        daily_task_limit: int,
        *,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.rate_limit_per_hour = rate_limit_per_hour
        self.daily_task_limit = daily_task_limit
        self.clock = clock or (lambda: datetime.now(SHANGHAI_TZ))
        self._hourly: Dict[str, Deque[float]] = defaultdict(deque)
        self._daily: Dict[date, int] = defaultdict(int)

    def reserve(self, client_ip: str) -> AdmissionReservation:
        now = self.clock().astimezone(SHANGHAI_TZ)
        timestamp = now.timestamp()
        hourly = self._hourly[client_ip]
        cutoff = timestamp - timedelta(hours=1).total_seconds()
        while hourly and hourly[0] <= cutoff:
            hourly.popleft()

        # Gate order is intentional: per-IP rate limit, then global daily limit.
        if len(hourly) >= self.rate_limit_per_hour:
            raise AdmissionError(429, "操作过于频繁，请稍后再试")
        if self._daily[now.date()] >= self.daily_task_limit:
            raise AdmissionError(429, "今日体验名额已用完，明天再来")

        hourly.append(timestamp)
        self._daily[now.date()] += 1
        self._discard_old_days(now.date())
        return AdmissionReservation(client_ip, timestamp, now.date())

    def rollback(self, reservation: AdmissionReservation) -> None:
        hourly = self._hourly.get(reservation.client_ip)
        if hourly is not None:
            try:
                hourly.remove(reservation.timestamp)
            except ValueError:
                pass
            if not hourly:
                self._hourly.pop(reservation.client_ip, None)
        remaining = self._daily.get(reservation.local_date, 0) - 1
        if remaining > 0:
            self._daily[reservation.local_date] = remaining
        else:
            self._daily.pop(reservation.local_date, None)

    def _discard_old_days(self, current_date: date) -> None:
        for day in list(self._daily):
            if day != current_date:
                self._daily.pop(day, None)
