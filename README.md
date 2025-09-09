# smart-classroom
# ==============================
# AI-Enhanced Career Guidance System (On-the-spot dataset)
# ==============================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === STEP 1: Enter your dataset manually ===
# Example dataset (you can replace with your own data anytime)
data = [
    {
        "title": "Data Scientist",
        "description": "Work with data, build ML models, and extract insights.",
        "skills": "python, statistics, machine learning, sql, pandas",
        "courses": "Python, Statistics, ML, Data Visualization",
        "salary_lpa": "12-40"
    },
    {
        "title": "Data Analyst",
        "description": "Analyze business data, build dashboards, and provide insights.",
        "skills": "excel, sql, tableau, powerbi, python",
        "courses": "Excel, SQL, Tableau, Business Analytics",
        "salary_lpa": "4-12"
    },
    {
        "title": "Cloud Engineer",
        "description": "Maintain cloud infrastructure, DevOps pipelines, and deployments.",
        "skills": "aws, azure, gcp, docker, kubernetes, linux",
        "courses": "AWS, Azure, DevOps, Docker, Kubernetes",
        "salary_lpa": "10-35"
    }
]

# Convert to DataFrame
df_careers = pd.DataFrame(data)

# Create corpus for vectorization
df_careers["corpus"] = df_careers.fillna("").apply(
    lambda r: " ".join([r["title"], r["description"], r["skills"], r["courses"]]), axis=1
)

# === STEP 2: Build TF-IDF Model ===
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
X_careers = vectorizer.fit_transform(df_careers["corpus"])

# === STEP 3: User Profile Class ===
class UserProfile:
    def __init__(self, name, education="", skills=None, interests=None, goals=""):
        self.name = name
        self.education = education
        self.skills = skills if skills else []
        self.interests = interests if interests else []
        self.goals = goals
    
    def to_text(self):
        return " ".join([self.education, " ".join(self.skills), " ".join(self.interests), self.goals])

# === STEP 4: Matching Functions ===
def profile_to_vector(profile: UserProfile):
    return vectorizer.transform([profile.to_text()])

def recommend_careers(profile: UserProfile, top_k=3):
    v = profile_to_vector(profile)
    sims = cosine_similarity(v, X_careers)[0]
    ranked = np.argsort(-sims)
    return [(df_careers.iloc[i].to_dict(), float(sims[i])) for i in ranked[:top_k]]

def match_skills(profile: UserProfile, career: dict):
    career_skills = [s.strip().lower() for s in str(career["skills"]).split(",")]
    user_skills = [s.lower() for s in profile.skills]
    return {
        "matched": [s for s in career_skills if s in user_skills],
        "missing": [s for s in career_skills if s not in user_skills]
    }

def generate_learning_pathway(profile: UserProfile, career: dict):
    analysis = match_skills(profile, career)
    steps = []
    for s in analysis["missing"][:5]:
        steps.append(f"Learn {s} through online courses")
    steps.append(f"Build 2 projects related to {career.get('title', 'career')} and publish on GitHub")
    steps.append("Update resume with new skills")
    return steps

# === STEP 5: Show Recommendations ===
def show_recommendations(profile: UserProfile, top_k=3):
    recs = recommend_careers(profile, top_k=top_k)
    print(f"\nCareer recommendations for {profile.name}:\n")
    for rank, (career, score) in enumerate(recs, 1):
        print(f"{rank}. {career['title']} (Match Score: {score:.2f})")
        print(f"   Description: {career['description']}")
        skills = match_skills(profile, career)
        print(f"   Matched skills: {', '.join(skills['matched']) or '(none)'}")
        print(f"   Missing skills: {', '.join(skills['missing']) or '(none)'}")
        print("   Suggested Pathway:")
        for step in generate_learning_pathway(profile, career):
            print(f"      - {step}")
        print(f"   Courses: {career['courses']}")
        print(f"   Salary (LPA): {career['salary_lpa']}")
        print("-" * 70)

# === STEP 6: Example Run ===
profile = UserProfile(
    name="Student A",
    education="B.Tech CSE",
    skills=["python", "sql", "excel"],
    interests=["machine learning", "cloud computing"],
    goals="Want a high-paying job in AI/ML"
)

show_recommendations(profile, top_k=3)
