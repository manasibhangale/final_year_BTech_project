class QuizState:
    def __init__(self):
        self.questions = []
        self.user_answers = {}
        self.score = 0
        self.weak_topics = []
        self.difficulty = "medium"
        self.current_index = 0

    def reset(self):
        self.questions = []
        self.user_answers = {}
        self.score = 0
        self.weak_topics = []
        self.current_index = 0