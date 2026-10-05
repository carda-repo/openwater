"""Dutch civil-time labels derived from the UTC instants used by features.

Use these helpers for clock hours, weekdays and activity dates. UTC instants
remain the basis for filtering, chronology, balances and elapsed durations.
Timezone-free day labels represent calendar dates, never elapsed timestamps.
"""

import pandas as pd


LOCAL_TIMEZONE = "Europe/Amsterdam"


def local_time(utc_timestamps: pd.Series) -> pd.Series:
    """Convert parsed timestamps to Dutch time, retaining summer/winter offsets.

    Existing feature readers strip the UTC zone after parsing. Such naive
    timestamps are UTC instants; attach UTC before converting, not Dutch time.
    """
    if utc_timestamps.dt.tz is None:
        utc_timestamps = utc_timestamps.dt.tz_localize("UTC")
    return utc_timestamps.dt.tz_convert(LOCAL_TIMEZONE)


def local_day_labels(utc_timestamps: pd.Series) -> pd.Series:
    """Naive midnight labels for Dutch calendar days, also safe on DST days."""
    return local_time(utc_timestamps).dt.tz_localize(None).dt.normalize()


def local_date(utc_timestamp: pd.Timestamp):
    """Dutch calendar date of one UTC instant, for calendar-day differences."""
    timestamp = pd.Timestamp(utc_timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert(LOCAL_TIMEZONE).date()
