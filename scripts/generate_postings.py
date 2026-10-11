"""Generate the SYNTHETIC job-postings dataset in data/postings.csv.

The sandbox that built this repo could not reach Kaggle or the Hugging Face Hub, so no
public postings dataset was available. This script is the fallback named in the project
spec: it writes ~500 clearly labelled synthetic postings from role-family templates.
The output is deterministic (fixed seed), so `--check` can verify the committed CSV.

Usage:
    python scripts/generate_postings.py            # write data/postings.csv
    python scripts/generate_postings.py --check    # regenerate in memory, compare bytes
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "postings.csv"
SEED = 20261011
N_POSTINGS = 500

# Each role family: title variants, the pool required skills are drawn from, a pool of
# nice-to-have skills, and a few sentences describing the day-to-day work.
ROLE_FAMILIES: dict[str, dict] = {
    "embedded_firmware": {
        "titles": [
            "Embedded Firmware Engineer",
            "Embedded Software Engineer",
            "Firmware Developer (MCU)",
            "Embedded C Developer",
        ],
        "required": [
            "c",
            "esp32",
            "stm32",
            "freertos",
            "uart",
            "i2c",
            "spi",
            "git",
            "oscilloscope",
        ],
        "nice": [
            "c++",
            "python",
            "bootloader",
            "low power design",
            "can bus",
            "pcb design",
            "ble",
            "unit testing",
            "autosar",
        ],
        "work": [
            "write bare-metal and RTOS firmware for microcontroller-based products",
            "bring up new boards, debug with oscilloscopes and logic analysers",
            "implement device drivers for sensors over UART, I2C and SPI",
            "own the firmware build, release and over-the-air update process",
            "work with hardware engineers on schematics and board bring-up",
        ],
    },
    "iot_engineer": {
        "titles": [
            "IoT Engineer",
            "IoT Solutions Engineer",
            "IoT Systems Developer",
            "Connected Devices Engineer",
        ],
        "required": [
            "mqtt",
            "esp32",
            "python",
            "c",
            "sensors",
            "tls",
            "aws iot core",
            "linux",
            "git",
        ],
        "nice": [
            "lorawan",
            "ble",
            "modbus",
            "node-red",
            "docker",
            "firebase",
            "monitoring",
            "zigbee",
            "low power design",
        ],
        "work": [
            "design the data path from sensor to cloud for fleets of connected devices",
            "provision devices with X.509 certificates and keep the fleet online",
            "build MQTT-based telemetry pipelines and dashboards",
            "integrate environmental sensors in agricultural and industrial settings",
            "troubleshoot connectivity in the field and on the broker side",
        ],
    },
    "data_analyst": {
        "titles": [
            "Data Analyst",
            "Business Data Analyst",
            "Analytics Engineer",
            "Junior Data Analyst",
        ],
        "required": [
            "sql",
            "python",
            "pandas",
            "excel",
            "data visualisation",
            "statistics",
            "power bi",
        ],
        "nice": ["airflow", "etl", "jupyter", "git", "spark", "time series", "english", "agile"],
        "work": [
            "turn raw operational data into dashboards and weekly reports",
            "write SQL against the warehouse and validate the numbers with stakeholders",
            "design A/B tests and explain the statistics to non-technical colleagues",
            "clean messy exports and document the data model",
            "automate recurring analyses with Python notebooks",
        ],
    },
    "ml_engineer": {
        "titles": ["Machine Learning Engineer", "ML Engineer", "Applied Scientist", "AI Engineer"],
        "required": [
            "python",
            "scikit-learn",
            "pytorch",
            "numpy",
            "pandas",
            "feature engineering",
            "mlops",
            "git",
        ],
        "nice": [
            "tensorflow",
            "docker",
            "aws",
            "nlp",
            "time series",
            "spark",
            "fastapi",
            "statistics",
            "unit testing",
        ],
        "work": [
            "train, evaluate and ship predictive models into production services",
            "build feature pipelines and monitor models after deployment",
            "run experiments and report results with proper validation splits",
            "package models as APIs and keep inference latency within budget",
            "collaborate with product teams to define what to predict and why",
        ],
    },
    "data_engineer": {
        "titles": ["Data Engineer", "Data Platform Engineer", "ETL Developer", "Big Data Engineer"],
        "required": ["python", "sql", "etl", "airflow", "spark", "docker", "aws", "git"],
        "nice": [
            "kubernetes",
            "terraform",
            "monitoring",
            "redis",
            "mongodb",
            "ci/cd",
            "linux",
            "pandas",
        ],
        "work": [
            "build and operate batch and streaming data pipelines",
            "model the warehouse and keep the data contracts stable",
            "scale ingestion on Spark and keep the cost down",
            "orchestrate jobs with Airflow and make failures visible",
            "own data quality checks and lineage",
        ],
    },
    "backend_python": {
        "titles": [
            "Backend Developer (Python)",
            "Python Developer",
            "Software Engineer, Backend",
            "API Developer",
        ],
        "required": [
            "python",
            "fastapi",
            "django",
            "rest apis",
            "sql",
            "docker",
            "git",
            "unit testing",
        ],
        "nice": [
            "redis",
            "websockets",
            "kubernetes",
            "ci/cd",
            "aws",
            "graphql",
            "mongodb",
            "linux",
            "agile",
        ],
        "work": [
            "design and build REST APIs that power web and mobile clients",
            "write well-tested Python services and review colleagues' code",
            "keep the database schema and migrations in good shape",
            "profile slow endpoints and fix them",
            "deploy services in containers with CI/CD",
        ],
    },
    "frontend": {
        "titles": ["Frontend Developer", "React Developer", "Web Developer", "UI Engineer"],
        "required": ["javascript", "typescript", "react", "html/css", "rest apis", "git"],
        "nice": ["node.js", "graphql", "selenium", "ci/cd", "websockets", "agile", "unit testing"],
        "work": [
            "build accessible, responsive interfaces in React and TypeScript",
            "integrate with backend APIs and handle loading and error states properly",
            "keep the component library consistent and documented",
            "write end-to-end tests for the critical user flows",
            "work with designers to turn mock-ups into working screens",
        ],
    },
    "mobile_flutter": {
        "titles": [
            "Flutter Developer",
            "Mobile Developer (Flutter)",
            "Mobile App Engineer",
            "Cross-platform Mobile Developer",
        ],
        "required": ["flutter", "dart", "firebase", "rest apis", "git"],
        "nice": ["android", "ios", "websockets", "ble", "unit testing", "ci/cd", "agile", "mqtt"],
        "work": [
            "build and publish cross-platform apps for Android and iOS",
            "connect apps to Firebase and REST backends with offline support",
            "implement real-time views of device data for IoT products",
            "keep app performance and battery use under control",
            "handle releases on both app stores",
        ],
    },
    "cloud_devops": {
        "titles": [
            "Cloud Engineer",
            "DevOps Engineer",
            "Site Reliability Engineer",
            "Platform Engineer",
            "AWS Solutions Engineer",
        ],
        "required": [
            "aws",
            "docker",
            "kubernetes",
            "terraform",
            "ci/cd",
            "linux",
            "bash",
            "monitoring",
            "networking",
        ],
        "nice": ["python", "azure", "gcp", "go", "redis", "tls", "git", "agile"],
        "work": [
            "run production infrastructure on AWS with infrastructure as code",
            "build CI/CD pipelines and keep deployments boring",
            "own monitoring, alerting and on-call runbooks",
            "harden networks, IAM policies and secrets handling",
            "reduce cloud spend without reducing reliability",
        ],
    },
    "computer_vision": {
        "titles": [
            "Computer Vision Engineer",
            "Vision Systems Engineer",
            "Deep Learning Engineer (Vision)",
            "Perception Engineer",
        ],
        "required": ["python", "opencv", "pytorch", "yolo", "numpy", "git"],
        "nice": [
            "tensorflow",
            "c++",
            "docker",
            "mlops",
            "linux",
            "esp32",
            "statistics",
            "feature engineering",
        ],
        "work": [
            "train and deploy object detectors for counting and inspection tasks",
            "collect and label image datasets and measure model quality honestly",
            "optimise models to run on edge hardware",
            "build camera pipelines with OpenCV",
            "evaluate detection quality with mAP and real failure cases",
        ],
    },
    "qa_test": {
        "titles": ["QA Engineer", "Test Automation Engineer", "Software Tester", "SDET"],
        "required": ["selenium", "python", "unit testing", "git", "agile", "rest apis"],
        "nice": ["javascript", "ci/cd", "sql", "docker", "linux", "technical writing", "java"],
        "work": [
            "design test plans and automate regression suites",
            "write API and UI tests and run them in CI",
            "report defects clearly and verify fixes",
            "measure test coverage and flakiness",
            "work with developers on testability",
        ],
    },
    "control_systems": {
        "titles": [
            "Control Systems Engineer",
            "Automation Engineer",
            "Controls Engineer",
            "Instrumentation and Control Engineer",
        ],
        "required": ["pid control", "matlab", "plc", "electronics", "signal processing", "sensors"],
        "nice": [
            "python",
            "c",
            "modbus",
            "thermodynamics",
            "stm32",
            "freertos",
            "can bus",
            "technical writing",
        ],
        "work": [
            "design and tune closed-loop controllers for thermal and motion systems",
            "model plants in MATLAB/Simulink and validate against measurements",
            "commission PLC and SCADA systems on site",
            "specify sensors and actuators and their safety interlocks",
            "write commissioning and test documentation",
        ],
    },
}

SENIORITY = [("Junior", 0, 0.28), ("", 1, 0.37), ("Senior", 4, 0.25), ("Lead", 6, 0.10)]

# All company names are fictional.
COMPANIES = [
    "Nile Delta Robotics",
    "Cairo Data Works",
    "Orchard Sense",
    "Pharos Cloud",
    "Sinai Systems",
    "Lotus Fields Tech",
    "Delta Grid Labs",
    "Aswan Analytics",
    "Red Sea Software",
    "Alexandria Vision",
    "Giza Embedded",
    "Mansoura Microsystems",
    "Suez Automation",
    "Luxor Mobile",
    "Sahara Edge",
    "Oasis Farms Digital",
    "Nubia Networks",
    "Tanta Telemetry",
    "Dakahlia Devices",
    "Zagazig Software",
    "Hamburg Sensorik GmbH",
    "Riyadh Smart Industries",
    "Dubai Fleet Systems",
    "Toronto Agri-Tech",
    "Osaka Control Works",
    "Budapest Data Studio",
    "Ankara Embedded",
    "Nairobi Climate Labs",
]
LOCATIONS = [
    ("Cairo, Egypt", 0.26),
    ("Giza, Egypt", 0.08),
    ("Alexandria, Egypt", 0.08),
    ("Mansoura, Egypt", 0.06),
    ("Remote", 0.22),
    ("Dubai, UAE", 0.08),
    ("Riyadh, Saudi Arabia", 0.06),
    ("Berlin, Germany", 0.06),
    ("Toronto, Canada", 0.04),
    ("Istanbul, Türkiye", 0.03),
    ("Budapest, Hungary", 0.03),
]
INTROS = [
    "We are looking for {a_title} to join our {team} team.",
    "{company} is hiring {a_title}.",
    "Join {company} as {a_title} and help us {mission}.",
    "{company} needs {a_title} for its {team} team.",
]
TEAMS = ["product", "platform", "R&D", "field operations", "data", "devices", "core engineering"]
MISSIONS = [
    "ship reliable products to farms and factories",
    "turn sensor data into decisions",
    "build software people depend on every day",
    "scale our platform to the next thousand customers",
    "bring modern tooling to a traditional industry",
]
EXTRAS = [
    "Hybrid work is possible after the first three months.",
    "You will present your work to the wider team every sprint.",
    "We value clear written communication and small, well-tested changes.",
    "Fresh graduates with strong project work are welcome to apply.",
    "A degree in computer, communications or electrical engineering is preferred.",
    "English is the working language; Arabic is a plus for customer visits.",
    "Expect occasional travel to customer sites.",
    "",
]
EDUCATION = {
    "Junior": "a bachelor's degree in engineering or computer science",
    "": "a bachelor's degree in engineering or computer science",
    "Senior": "a bachelor's or master's degree in a relevant field",
    "Lead": "a master's degree or equivalent experience",
}


def _weighted(rng: random.Random, options: list[tuple]) -> tuple:
    weights = [o[-1] for o in options]
    return rng.choices(options, weights=weights, k=1)[0]


def _article(noun: str) -> str:
    return ("an " if noun[0].lower() in "aeiou" else "a ") + noun


def _skill_phrase(skills: list[str]) -> str:
    # Mild variation so the text does not read like a list dump.
    return ", ".join(skills[:-1]) + f" and {skills[-1]}" if len(skills) > 1 else skills[0]


def make_posting(rng: random.Random, pid: int) -> dict:
    family = rng.choice(list(ROLE_FAMILIES))
    spec = ROLE_FAMILIES[family]
    level, years_base, _ = _weighted(rng, SENIORITY)
    base_title = rng.choice(spec["titles"])
    title = f"{level} {base_title}".strip()
    n_req = rng.randint(4, min(6, len(spec["required"])))
    required = rng.sample(spec["required"], n_req)
    nice = rng.sample(spec["nice"], rng.randint(1, 4))
    years_min = years_base + rng.choice([0, 0, 1])
    company = rng.choice(COMPANIES)
    location, _ = _weighted(rng, LOCATIONS)
    remote = location == "Remote" or rng.random() < 0.15
    work = rng.sample(spec["work"], 3)
    intro = rng.choice(INTROS).format(
        a_title=_article(title),
        company=company,
        team=rng.choice(TEAMS),
        mission=rng.choice(MISSIONS),
    )
    desc = [
        intro,
        f"You will {work[0]}, {work[1]} and {work[2]}.",
        f"Required: {_skill_phrase(required)}.",
        f"Nice to have: {_skill_phrase(nice)}.",
        (
            f"We expect at least {years_min} year{'s' if years_min > 1 else ''} of relevant "
            f"experience and {EDUCATION[level]}."
            if years_min
            else f"No prior professional experience is required; {EDUCATION[level]} is expected."
        ),
        rng.choice(EXTRAS),
    ]
    return {
        "id": f"P{pid:04d}",
        "title": title,
        "company": company,
        "location": location,
        "remote": str(remote).lower(),
        "seniority": level or "Mid",
        "years_min": years_min,
        "role_family": family,
        "required_skills": ";".join(required),
        "nice_to_have": ";".join(nice),
        "description": " ".join(s for s in desc if s),
    }


FIELDS = [
    "id",
    "title",
    "company",
    "location",
    "remote",
    "seniority",
    "years_min",
    "role_family",
    "required_skills",
    "nice_to_have",
    "description",
]


def generate(seed: int = SEED, n: int = N_POSTINGS) -> str:
    rng = random.Random(seed)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for i in range(1, n + 1):
        writer.writerow(make_posting(rng, i))
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify data/postings.csv is current")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("-n", type=int, default=N_POSTINGS)
    args = parser.parse_args(argv)
    text = generate(args.seed, args.n)
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT} is out of date; run `make data`", file=sys.stderr)
            return 1
        print(f"{OUT}: up to date ({args.n} postings)")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({args.n} postings, seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
