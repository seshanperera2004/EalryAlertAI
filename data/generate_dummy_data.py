import random
import csv
import os

random.seed(42)  # reproducible dataset

NUM_STUDENTS = 600
FACULTIES = [
    "FMSH", "FOE", "FOL", "FOM", "FDSS",
    "FOT", "FOCJ", "FBESS", "FOC", "FAHS",
]
YEARS = [1, 2, 3, 4]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "students_dummy.csv")


def clamp(value, low, high):
    return max(low, min(high, value))


def generate_student(student_id):
    """
    Generate one synthetic student record.

    A hidden 'true_risk_tier' drives correlated feature generation so the
    dataset has real learnable signal (not pure noise) — this mirrors how
    genuinely at-risk students tend to show several warning signs at once,
    not just one in isolation.
    """
    true_risk = random.choices(
        ["low", "moderate", "high"], weights=[0.62, 0.26, 0.12], k=1
    )[0]

    faculty = random.choice(FACULTIES)
    year = random.choice(YEARS)
    gender = random.choice(["Male", "Female"])

    # Base ("healthy") ranges, shifted upward the higher the true risk tier is
    if true_risk == "low":
        attendance_decline = random.gauss(5, 4)
        lms_inactivity = random.gauss(2, 1.5)
        delay_index = random.gauss(0.1, 0.1)
        volatility = random.gauss(5, 3)
    elif true_risk == "moderate":
        attendance_decline = random.gauss(22, 8)
        lms_inactivity = random.gauss(7, 3)
        delay_index = random.gauss(0.35, 0.15)
        volatility = random.gauss(15, 5)
    else:  # high
        attendance_decline = random.gauss(45, 12)
        lms_inactivity = random.gauss(14, 4)
        delay_index = random.gauss(0.65, 0.2)
        volatility = random.gauss(28, 8)

    # Add independent noise so features aren't perfectly correlated
    # (real students are messy — someone can miss class but still submit on time)
    attendance_decline += random.gauss(0, 6)
    lms_inactivity += random.gauss(0, 2)
    delay_index += random.gauss(0, 0.08)
    volatility += random.gauss(0, 4)

    attendance_decline = round(clamp(attendance_decline, 0, 100), 1)
    lms_inactivity = round(clamp(lms_inactivity, 0, 30), 1)
    delay_index = round(clamp(delay_index, 0, 1), 2)
    volatility = round(clamp(volatility, 0, 40), 1)

    # Current average mark: inversely related to risk, plus noise
    base_avg = {"low": 72, "moderate": 58, "high": 44}[true_risk]
    current_avg_mark = round(clamp(random.gauss(base_avg, 9), 20, 100), 1)

    # A small fraction of "surprise" cases: good grades but rising disengagement
    # (this is exactly the kind of case EarlyAlertAI is meant to catch early)
    is_hidden_risk = random.random() < 0.05
    if is_hidden_risk:
        current_avg_mark = round(clamp(random.gauss(68, 5), 40, 100), 1)
        lms_inactivity = round(clamp(lms_inactivity + random.uniform(6, 10), 0, 30), 1)
        attendance_decline = round(clamp(attendance_decline + random.uniform(15, 25), 0, 100), 1)

    return {
        "student_id": f"KDU{10000 + student_id}",
        "faculty": faculty,
        "year_of_study": year,
        "gender": gender,
        "attendance_decline_pct": attendance_decline,
        "lms_inactivity_gap_days": lms_inactivity,
        "assignment_delay_index": delay_index,
        "grade_volatility_score": volatility,
        "current_avg_mark": current_avg_mark,
        # kept ONLY for generating a realistic label during model training;
        # a real deployment would never have this ground-truth column up front
        "true_risk_tier": true_risk,
    }


def main():
    rows = [generate_student(i) for i in range(NUM_STUDENTS)]

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {NUM_STUDENTS} synthetic student records -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
