import os
import sqlite3
import pandas as pd
import datetime
from pathlib import Path

class AttendanceDB:
    def __init__(self, db_path=None):
        """Initialize the attendance database"""
        if db_path is None:
            # Use default location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(current_dir, "attendance.db")
        
        self.db_path = db_path
        self.conn = None
        self.initialize_db()
    
    def get_connection(self):
        """Get a database connection"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn
    
    def close_connection(self):
        """Close the database connection"""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
    
    def initialize_db(self):
        """Initialize the database schema if it doesn't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create persons table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY,
            name TEXT,
            person_id TEXT UNIQUE,
            registered_date TEXT
        )
        ''')
        
        # Create attendance table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY,
            person_id INTEGER,
            date TEXT,
            time TEXT,
            status INTEGER,
            FOREIGN KEY (person_id) REFERENCES persons (id),
            UNIQUE(person_id, date)
        )
        ''')
        
        conn.commit()
    
    def register_person(self, person_id, name=None):
        """Register a new person or update existing person"""
        if name is None:
            name = f"Person {person_id}"
            
        conn = self.get_connection()
        cursor = conn.cursor()
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        try:
            cursor.execute('''
            INSERT OR IGNORE INTO persons (person_id, name, registered_date)
            VALUES (?, ?, ?)
            ''', (person_id, name, today))
            
            # If no insertion happened, update the record
            if cursor.rowcount == 0:
                cursor.execute('''
                UPDATE persons SET name = ? WHERE person_id = ?
                ''', (name, person_id))
                
            conn.commit()
            return self.get_person_by_id(person_id)
        except Exception as e:
            conn.rollback()
            print(f"Error registering person: {e}")
            return None
    
    def get_person_by_id(self, person_id):
        """Get person by external ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, name, person_id, registered_date FROM persons WHERE person_id = ?
        ''', (person_id,))
        
        result = cursor.fetchone()
        if result:
            return {
                "id": result[0],
                "name": result[1],
                "person_id": result[2],
                "registered_date": result[3]
            }
        return None
    
    def mark_attendance(self, person_id, status=1, date=None, time=None):
        """Mark attendance for a person
        
        Args:
            person_id (str): The person's ID
            status (int): 1 for present, 0 for absent
            date (str, optional): Date in YYYY-MM-DD format. Defaults to today.
            time (str, optional): Time in HH:MM format. Defaults to current time.
        """
        # Get database person record or register if not exists
        person = self.get_person_by_id(person_id)
        if not person:
            person = self.register_person(person_id)
            if not person:
                return False
        
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        if time is None:
            time = datetime.datetime.now().strftime("%H:%M:%S")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT OR REPLACE INTO attendance (person_id, date, time, status)
            VALUES (?, ?, ?, ?)
            ''', (person["id"], date, time, status))
            
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            print(f"Error marking attendance: {e}")
            return False
    
    def get_attendance_by_date(self, date=None):
        """Get attendance records for a specific date"""
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        conn = self.get_connection()
        
        query = '''
        SELECT p.id, p.name, p.person_id, a.date, a.time, a.status
        FROM persons p
        LEFT JOIN attendance a ON p.id = a.person_id AND a.date = ?
        ORDER BY p.id
        '''
        
        df = pd.read_sql_query(query, conn, params=(date,))
        return df
    
    def get_attendance_range(self, start_date, end_date):
        """Get attendance records for a date range"""
        conn = self.get_connection()
        
        # First get all persons
        persons = pd.read_sql_query("SELECT id, name, person_id FROM persons", conn)
        
        # Get all dates in range
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        date_range = [(start + datetime.timedelta(days=x)).strftime("%Y-%m-%d") 
                     for x in range((end - start).days + 1)]
        
        # Get attendance for all dates in range
        attendance = pd.read_sql_query('''
        SELECT person_id, date, status 
        FROM attendance 
        WHERE date BETWEEN ? AND ?
        ''', conn, params=(start_date, end_date))
        
        # Create a pivot table with dates as columns and person IDs as rows
        if not attendance.empty:
            pivot = attendance.pivot(index='person_id', columns='date', values='status')
            
            # Merge with persons data
            result = persons.join(pivot, on='id', how='left')
            
            # Fill missing values with 0 (absent by default)
            for date in date_range:
                if date not in result.columns:
                    result[date] = 0
                else:
                    result[date] = result[date].fillna(0).astype(int)
            
            return result
        else:
            # If no attendance data, create empty dataframe with all dates
            for date in date_range:
                persons[date] = 0
            return persons
    
    def export_attendance_to_excel(self, start_date=None, end_date=None, output_path=None):
        """Export attendance to Excel file"""
        # Set default dates to current month
        if start_date is None or end_date is None:
            today = datetime.datetime.now()
            first_day = datetime.datetime(today.year, today.month, 1)
            
            # Last day of current month
            if today.month == 12:
                last_day = datetime.datetime(today.year, 12, 31)
            else:
                last_day = datetime.datetime(today.year, today.month + 1, 1) - datetime.timedelta(days=1)
            
            start_date = first_day.strftime("%Y-%m-%d")
            end_date = last_day.strftime("%Y-%m-%d")
        
        # Set default output path
        if output_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            filename = f"attendance_{start_date}_to_{end_date}.xlsx"
            output_path = os.path.join(current_dir, filename)
        
        # Get attendance data
        attendance_df = self.get_attendance_range(start_date, end_date)
        
        if attendance_df.empty:
            print("No attendance records to export")
            return None
        
        # Create a writer to save the Excel file
        writer = pd.ExcelWriter(output_path, engine='openpyxl')
        
        # Remove database IDs from the output
        if 'id' in attendance_df.columns:
            output_df = attendance_df.drop('id', axis=1)
        else:
            output_df = attendance_df.copy()
            
        # Rename columns for clarity
        output_df = output_df.rename(columns={
            'person_id': 'Person ID',
            'name': 'Name'
        })
        
        # Add summary column for attendance count
        date_columns = [col for col in output_df.columns if col not in ['Person ID', 'Name']]
        if date_columns:
            output_df['Total Present'] = output_df[date_columns].sum(axis=1)
            output_df['Attendance %'] = (output_df['Total Present'] / len(date_columns) * 100).round(2)
            
        # Write to Excel
        output_df.to_excel(writer, sheet_name='Attendance', index=False)
        
        # Save the workbook without attempting to adjust column widths
        writer.close()
        
        return output_path

# Testing function
if __name__ == "__main__":
    db = AttendanceDB()
    # Register some test persons
    for i in range(1, 5):
        db.register_person(str(i), f"Test Person {i}")
    
    # Mark attendance for today
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    db.mark_attendance("1", 1, today)  # Person 1 present
    db.mark_attendance("2", 0, today)  # Person 2 absent
    
    # Get attendance for today
    print(db.get_attendance_by_date(today))
    
    # Export to Excel
    excel_path = db.export_attendance_to_excel()
    print(f"Exported attendance to {excel_path}")