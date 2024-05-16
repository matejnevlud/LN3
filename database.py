import sqlite3
from datetime import date


# create a database class for storing camera detection events, with id of region, date of detection and counts of detections on that date
class Database:
    def __init__(self):
        self.conn = sqlite3.connect('detections.db')
        self.cursor = self.conn.cursor()
        # create table with self incrementing primary key id,
        # region_id, date and count of detections on that date
        self.cursor.execute('CREATE TABLE IF NOT EXISTS detections (id INTEGER PRIMARY KEY AUTOINCREMENT, region_id INTEGER, date TEXT, count INTEGER)')

        # create view, which will show sum of detections for each date
        self.cursor.execute('CREATE VIEW IF NOT EXISTS detections_by_date AS SELECT date, SUM(count) FROM detections GROUP BY date')

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

    def close(self):
        self.conn.close()