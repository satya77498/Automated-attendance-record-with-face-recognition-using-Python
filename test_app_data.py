import datetime as dt
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train


class AppDataTests(unittest.TestCase):
    def test_parse_cutoff_time_validates_common_case(self):
        self.assertEqual(train.parse_cutoff_time("09:15"), dt.time(9, 15))
        self.assertEqual(train.parse_cutoff_time("invalid", "10:30"), dt.time(10, 30))

    def test_attendance_summary_returns_zeroes_for_missing_day(self):
        summary = train.build_attendance_summary(train.ATTENDANCE_DIR / "Attendance_2099-01-01.csv")
        self.assertEqual(summary["present"], 0)
        self.assertEqual(summary["late"], 0)
        self.assertEqual(summary["absent"], 0)
        self.assertEqual(summary["total"], 0)

    def test_load_users_handles_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_file = Path(directory) / "students.csv"
            with patch.object(train, "STUDENT_FILE", missing_file):
                self.assertEqual(train.load_users(), {})

    def test_read_csv_rows_refreshes_cache_after_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "attendance.csv"
            csv_path.write_text("Id,Name\n1,Ada\n", encoding="utf-8")
            first_read = train.read_csv_rows(csv_path)
            csv_path.write_text("Id,Name\n1,Ada\n2,Lin\n", encoding="utf-8")

            second_read = train.read_csv_rows(csv_path)

        self.assertEqual(len(first_read), 1)
        self.assertEqual(len(second_read), 2)

    def test_attendance_dashboard_uses_registered_people_and_on_time_status(self):
        users = {"7": "Ada"}
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "attendance.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(train.ATTENDANCE_COLUMNS)
                writer.writerow(("07", "Incorrect CSV name", "2026-09-26", "09:00:00", "On time"))
                writer.writerow(("7", "Ada", "2026-09-26", "09:10:00", "Late"))
                writer.writerow(("99", "Unknown", "2026-09-26", "09:15:00", "Present"))

            rows = train.registered_attendance_rows(train.read_csv_rows(csv_path), users)
            summary = train.build_attendance_summary(csv_path, registered_users=users)
            empty_roster_summary = train.build_attendance_summary(csv_path, registered_users={})

        self.assertEqual([row["Name"] for row in rows], ["Ada", "Ada"])
        self.assertEqual(summary["present"], 1)
        self.assertEqual(summary["late"], 1)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(empty_roster_summary["total"], 0)

    def test_weekly_chart_places_saturday_count_on_saturday(self):
        class RecordingCanvas:
            def __init__(self):
                self.items = []

            def delete(self, _tag):
                self.items.clear()

            def winfo_width(self):
                return 700

            def winfo_height(self):
                return 130

            def create_line(self, *args, **kwargs):
                self.items.append(("line", args, kwargs))

            def create_rectangle(self, *args, **kwargs):
                self.items.append(("rectangle", args, kwargs))

            def create_text(self, *args, **kwargs):
                self.items.append(("text", args, kwargs))

        canvas = RecordingCanvas()
        app = train.AttendanceApp.__new__(train.AttendanceApp)
        app._draw_weekly_bars(
            canvas,
            [dt.date(2026, 9, 25), dt.date(2026, 9, 26)],
            [0, 1],
            [0, 1],
        )

        text_items = [item for item in canvas.items if item[0] == "text"]
        day_positions = {
            item[2]["text"]: item[1][0]
            for item in text_items
            if item[2]["text"] in {"Fri", "Sat"}
        }
        count_positions = [
            item[1][0] for item in text_items if item[2]["text"] == "1"
        ]
        bars = [item for item in canvas.items if item[0] == "rectangle"]

        self.assertLess(day_positions["Fri"], day_positions["Sat"])
        self.assertEqual(count_positions, [day_positions["Sat"]])
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0][2]["fill"], train.COLORS["coral"])

    def test_attendance_period_bounds_cover_week_and_month(self):
        anchor = dt.date(2026, 9, 30)

        self.assertEqual(
            train.attendance_period_bounds("week", anchor),
            (dt.date(2026, 9, 28), dt.date(2026, 10, 4)),
        )
        self.assertEqual(
            train.attendance_period_bounds("month", anchor),
            (dt.date(2026, 9, 1), dt.date(2026, 9, 30)),
        )

    def test_find_attendance_proof_matches_person_and_date(self):
        with tempfile.TemporaryDirectory() as directory:
            proof_root = Path(directory)
            proof_day = proof_root / "2026-09-26"
            proof_day.mkdir()
            first = proof_day / "7_090000_000001.jpg"
            latest = proof_day / "7_090000_000002.jpg"
            other_person = proof_day / "8_090000_000003.jpg"
            for path in (first, latest, other_person):
                path.touch()

            with patch.object(train, "ATTENDANCE_PROOF_DIR", proof_root):
                self.assertEqual(
                    train.find_attendance_proof("007", "2026-09-26"), latest,
                )
                self.assertIsNone(train.find_attendance_proof("7", "2026-09-25"))

    def test_duplicate_vote_rejects_face_despite_noisy_frame(self):
        votes = {}
        predictions = [("12", 35), ("12", 60), ("12", 38), ("12", 40)]

        decisions = [
            train._record_duplicate_vote(votes, person_id, confidence, 45)
            for person_id, confidence in predictions
        ]

        self.assertEqual(decisions, [False, False, False, True])

    def test_duplicate_votes_are_counted_per_enrolled_profile(self):
        votes = {}

        decisions = [
            train._record_duplicate_vote(votes, person_id, 35, 45)
            for person_id in ("12", "18", "12", "18")
        ]

        self.assertEqual(decisions, [False, False, False, False])


if __name__ == "__main__":
    unittest.main()
