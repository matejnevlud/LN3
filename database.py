import sqlite3
import time
from datetime import date



class Database:
    def __init__(self):
        self.conn = sqlite3.connect('detections.db')
        self.cursor = self.conn.cursor()
        # create table with self incrementing primary key id,
        # region_id, date and count of detections on that date
        self.cursor.execute('CREATE TABLE IF NOT EXISTS detections (id INTEGER PRIMARY KEY AUTOINCREMENT, region_id INTEGER, date TEXT, count INTEGER)')

        #also create table for logging start and stop of movement, with timestamp
        self.cursor.execute('CREATE TABLE IF NOT EXISTS movement (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, is_moving INTEGER)')

        # create view, which will show sum of detections for each date
        self.cursor.execute('CREATE VIEW IF NOT EXISTS detections_by_date AS SELECT date, SUM(count) FROM detections GROUP BY date')

        # create view for movements calculation
        self.cursor.execute("CREATE VIEW IF NOT EXISTS movement_diff AS WITH previous_moving AS (SELECT id, timestamp, is_moving, LAG(timestamp) OVER (ORDER BY CAST(timestamp AS REAL)) AS prev_moving_timestamp FROM movement WHERE is_moving = 1), all_rows_with_prev_moving AS (SELECT m.id, m.timestamp AS stop_timestamp, m.is_moving, (SELECT timestamp FROM previous_moving pm WHERE CAST(pm.timestamp AS REAL) <= CAST(m.timestamp AS REAL) ORDER BY CAST(pm.timestamp AS REAL) DESC LIMIT 1) AS start_timestamp FROM movement m) SELECT id, start_timestamp, stop_timestamp, DATE(stop_timestamp, 'unixepoch') AS date, (CAST(stop_timestamp AS REAL) - CAST(start_timestamp AS REAL)) / 60.0 AS diff_in_minutes FROM all_rows_with_prev_moving WHERE is_moving = 0 ORDER BY CAST(stop_timestamp AS REAL);")

    def add_detection(self, region_id):
        # date in format YYYY-MM-DD
        current_date = date.today().isoformat()
        self.cursor.execute('SELECT count FROM detections WHERE region_id = ? AND date = ?', (region_id, current_date))
        row = self.cursor.fetchone()
        if row is None:
            self.cursor.execute('INSERT INTO detections (region_id, date, count) VALUES (?, ?, 1)', (region_id, current_date))
        else:
            self.cursor.execute('UPDATE detections SET count = count + 1 WHERE region_id = ? AND date = ?', (region_id, current_date))
        self.conn.commit()

    def get_detections(self):
        self.cursor.execute('SELECT * FROM detections')
        return self.cursor.fetchall()

    def get_detections_by_date(self, date_iso):
        self.cursor.execute('SELECT * FROM detections WHERE date = ?', (date_iso,))
        return self.cursor.fetchall()

    def add_movement(self, _is_moving):
        self.cursor.execute('INSERT INTO movement (timestamp, is_moving) VALUES (?, ?)', (time.time(), 1 if _is_moving else 0))
        self.conn.commit()


    def close(self):
        self.conn.close()
