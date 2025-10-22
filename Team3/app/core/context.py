import os
from core.hailo_cam_adapter import HailoCamAdapter  
from core.face_service import FaceService
from db.database import create_engine_and_session, init_db
from db.models import WorkoutSession, SessionExercise

class AppContext:
    def __init__(self):
        root = os.path.dirname(os.path.dirname(__file__))  
        data_root = os.path.join(root, "data")
        os.makedirs(data_root, exist_ok=True)

        db_path = os.path.join(data_root, "app.db")
        self.engine, self.SessionLocal = create_engine_and_session(db_path)
        init_db(self.engine)

        self.face = FaceService(self.SessionLocal)

        self.cam = HailoCamAdapter()

        self.router = None
        self.current_exercise = None

        self.current_user_id: int | None = None
        self.current_user_name: str | None = None
        
    def set_router(self, router):
        self.router = router

    def goto_summary(self, summary: dict):
        if not self.router:
            return
        page = self.router.navigate("summary")
        if hasattr(page, "set_data"):
            page.set_data(summary)

    def goto_main(self):
        if self.router:
            self.router.navigate("select")

    def goto_profile(self):
        if self.router:
            self.router.navigate("info")

    def restart_current_exercise(self, ex: str | None = None):
        if self.router:
            self.router.navigate("exercise") 

    def set_current_user(self, user_id: int, name: str):
        self.current_user_id = user_id
        self.current_user_name = name

    def clear_current_user(self):
        self.current_user_id = None
        self.current_user_name = None

    def is_logged_in(self) -> bool:
        return self.current_user_id is not None
    
    def save_workout_session(self, summary: dict):
        if not self.is_logged_in():
            return

        duration_sec = int(summary.get("duration_sec", 0))
        avg_score = float(summary.get("avg_score", 0.0))
        exercises = list(summary.get("exercises") or [])

        with self.SessionLocal() as s:
            sess = WorkoutSession(
                user_id=self.current_user_id,
                duration_sec=duration_sec,
                avg_score=avg_score,
            )
            s.add(sess)
            s.flush()  

            for idx, item in enumerate(exercises):
                name = str(item.get("name", "운동")).strip()
                reps = int(item.get("reps", 0))
                avg  = float(item.get("avg", item.get("avg_score", 0.0)))

                s.add(SessionExercise(
                    session_id=sess.id,
                    exercise_name=name,
                    reps=reps,
                    avg_score=avg,
                    order_index=idx
                ))
            s.commit()
