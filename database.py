import sqlite3
import time
from datetime import date



class Database:
    def __init__(self):
        self.conn = sqlite3.connect('detections.db')
        self.cursor = self.conn.cursor()
        # create table with self incrementing primary key id,
        # region_id, date and count of detections on that date
        self.cursor.execute('CREATE TABLE IF NOT EXISTS measurements (id INTEGER PRIMARY KEY AUTOINCREMENT, empty INTEGER, filled INTEGER, date TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
        self.cursor.execute('CREATE VIEW IF NOT EXISTS daily_sums AS SELECT date, SUM(empty) AS sum_empty, SUM(filled) AS sum_filled, SUM(empty + filled) AS sum_total FROM measurements GROUP BY date')

        self.cursor.execute('CREATE TABLE IF NOT EXISTS movement (id INTEGER PRIMARY KEY AUTOINCREMENT, is_moving INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
        self.cursor.execute("CREATE VIEW IF NOT EXISTS movement_diff AS WITH previous_moving AS (SELECT id, timestamp, is_moving, LAG(timestamp) OVER (ORDER BY timestamp) AS prev_moving_timestamp FROM movement WHERE is_moving = 1), all_rows_with_prev_moving AS (SELECT m.id, m.timestamp AS stop_timestamp, m.is_moving, (SELECT timestamp FROM previous_moving pm WHERE pm.timestamp <= m.timestamp ORDER BY pm.timestamp DESC LIMIT 1) AS start_timestamp FROM movement m) SELECT id, start_timestamp, stop_timestamp, DATE(stop_timestamp) AS date, (JULIANDAY(stop_timestamp) - JULIANDAY(start_timestamp)) * 1440.0 AS diff_in_minutes FROM all_rows_with_prev_moving WHERE is_moving = 0 ORDER BY stop_timestamp;")




        self.cursor.execute('CREATE TABLE IF NOT EXISTS detections (id INTEGER PRIMARY KEY AUTOINCREMENT, region_id INTEGER, date TEXT, count INTEGER)')

        #also create table for logging start and stop of movement, with timestamp

        # create view, which will show sum of detections for each date
        self.cursor.execute('CREATE VIEW IF NOT EXISTS detections_by_date AS SELECT date, SUM(count) FROM detections GROUP BY date')

        # create view for movements calculation

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
        self.cursor.execute('INSERT INTO movement (is_moving) VALUES (?)', (_is_moving,))
        self.conn.commit()


    def add_measurement(self, empty, filled):
        current_date = date.today().isoformat()
        self.cursor.execute('INSERT INTO measurements (empty, filled, date) VALUES (?, ?, ?)', (empty, filled, current_date))
        self.conn.commit()




    def save_conveyer_measurements(self, conveyer):
        measurements = conveyer.get_avg_detection()
        empty = len([measurement for measurement in measurements if not measurement])
        filled = len([measurement for measurement in measurements if measurement])
        self.add_measurement(empty, filled)
        print(''.join(['● ' if measurement else '◯ ' for measurement in measurements]))

    def close(self):
        self.conn.close()
