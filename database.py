from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Violation(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    video_name = db.Column(db.String(200))

    violation_type = db.Column(db.String(50))

    evidence_image = db.Column(db.String(200))

    plate_image = db.Column(db.String(200))

    processing_time = db.Column(db.Float)

    date_time = db.Column(db.String(100))

    status = db.Column(db.String(50), default="unreviewed")

    def __repr__(self):
        return f"<Violation {self.id}>"
