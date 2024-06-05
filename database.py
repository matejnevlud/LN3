import sqlite3
import time
from datetime import date, datetime, timedelta


class Database:
    def __init__(self):
        self.conn = sqlite3.connect('detections.db')
        self.cursor = self.conn.cursor()
        # create table with self incrementing primary key id,
        # region_id, date and count of detections on that date
        self.cursor.execute('CREATE TABLE IF NOT EXISTS measurements (id INTEGER PRIMARY KEY AUTOINCREMENT, empty INTEGER, filled INTEGER, date TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
        self.cursor.execute('CREATE VIEW IF NOT EXISTS daily_sums AS SELECT date, SUM(empty) AS sum_empty, SUM(filled) AS sum_filled, SUM(empty + filled) AS sum_total FROM measurements GROUP BY date')

        self.cursor.execute('CREATE TABLE IF NOT EXISTS movement (id INTEGER PRIMARY KEY AUTOINCREMENT, is_moving INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
        self.cursor.execute(
            "CREATE VIEW IF NOT EXISTS movement_diff AS WITH previous_moving AS (SELECT id, timestamp, is_moving, LAG(timestamp) OVER (ORDER BY timestamp) AS prev_moving_timestamp FROM movement WHERE is_moving = 1), all_rows_with_prev_moving AS (SELECT m.id, m.timestamp AS stop_timestamp, m.is_moving, (SELECT timestamp FROM previous_moving pm WHERE pm.timestamp <= m.timestamp ORDER BY pm.timestamp DESC LIMIT 1) AS start_timestamp FROM movement m) SELECT id, start_timestamp, stop_timestamp, DATE(stop_timestamp) AS date, (JULIANDAY(stop_timestamp) - JULIANDAY(start_timestamp)) * 1440.0 AS diff_in_minutes FROM all_rows_with_prev_moving WHERE is_moving = 0 ORDER BY stop_timestamp;")

        self.cursor.execute('CREATE TABLE IF NOT EXISTS detections (id INTEGER PRIMARY KEY AUTOINCREMENT, region_id INTEGER, date TEXT, count INTEGER)')

        #also create table for logging start and stop of movement, with timestamp

        # create view, which will show sum of detections for each date
        self.cursor.execute('CREATE VIEW IF NOT EXISTS detections_by_date AS SELECT date, SUM(count) FROM detections GROUP BY date')

        self.no_empty = 0
        self.no_filled = 0
        self.date = None
        self.time_of_movement = None

        self.is_moving = False
        self.last_movement_change = time.time()

    def add_movement(self, _is_moving):
        current_datetime = time.time()
        self.cursor.execute('INSERT INTO movement (is_moving) VALUES (?)', (_is_moving,))
        self.conn.commit()

        if self.last_movement_change is None:
            self.last_movement_change = current_datetime
            self.is_moving = _is_moving
        elif self.is_moving != _is_moving:
            self.is_moving = _is_moving
            self.last_movement_change = current_datetime

    def get_last_movement(self):
        self.cursor.execute('SELECT is_moving, timestamp FROM movement ORDER BY timestamp DESC LIMIT 1')
        row = self.cursor.fetchone()
        if row is not None:
            self.is_moving = row[0] == 1
            # row[1] is in format 'YYYY-MM-DD HH:MM:SS' parse to time object, time is in UTC

            self.last_movement_change = time.mktime((datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S') - timedelta(hours=int(-2))).timetuple())
        return self.is_moving



    def add_measurement(self, empty, filled):
        current_date = date.today().isoformat()
        self.cursor.execute('INSERT INTO measurements (empty, filled, date) VALUES (?, ?, ?)', (empty, filled, current_date))
        self.conn.commit()

        if self.date is None:
            self.date = current_date
            # try to get last measurements from database
            self.cursor.execute('SELECT sum_empty, sum_filled FROM daily_sums WHERE date = ?', (current_date,))
            row = self.cursor.fetchone()
            if row is not None:
                self.no_empty = row[0]
                self.no_filled = row[1]
            else:
                self.no_empty = empty
                self.no_filled = filled
        elif self.date == current_date:
            self.no_empty += empty
            self.no_filled += filled
        elif self.date != current_date:
            self.date = current_date
            self.no_empty = empty
            self.no_filled = filled

    def save_conveyer_measurements(self, conveyer):
        measurements = conveyer.get_avg_detection()
        empty = len([measurement for measurement in measurements if not measurement])
        filled = len([measurement for measurement in measurements if measurement])
        self.add_measurement(empty, filled)
        print(''.join(['● ' if measurement else '◯ ' for measurement in measurements]))

    def close(self):
        self.conn.close()
