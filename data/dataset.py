import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

QUESTION_TEMPLATES = {
    "birthdate": "When was {first_name} {last_name} born?",
    "birthplace": "Where was {first_name} {last_name} born?",
    "employer": "Where does {first_name} {last_name} work?",
    "university": "What university did {first_name} {last_name} attend?",
}


@dataclass(frozen=True)
class Profile:
    identifier: int
    first_name: str
    last_name: str
    birthdate: datetime
    birthplace: str
    employer: str
    university: str


def generate_profiles(seed: int = 0, path="data"):
    prng = random.Random(seed)
    path = path + "/names/"

    with open(path + "first_name.txt", "r") as f:
        first_names = [line.strip() for line in f]
        prng.shuffle(first_names)

    with open(path + "last_name.txt", "r") as f:
        last_names = [line.strip() for line in f]
        prng.shuffle(last_names)

    with open(path + "employer.txt", "r") as f:
        employers = [line.strip() for line in f]
        prng.shuffle(employers)

    with open(path + "town.txt", "r") as f:
        towns = [line.strip() for line in f]
        prng.shuffle(towns)

    with open(path + "university.txt", "r") as f:
        universities = [line.strip() for line in f]
        prng.shuffle(universities)

    for i, (first, last) in enumerate(zip(first_names, last_names)):
        start = datetime(1995, 7, 2)
        end = datetime(2000, 2, 14)
        timespan = end - start

        random_days = prng.randint(0, timespan.days)
        birthdate = start + timedelta(days=random_days)
        yield Profile(
            identifier=i,
            first_name=first,
            last_name=last,
            birthdate=birthdate,
            employer=prng.choice(employers),
            birthplace=prng.choice(towns),
            university=prng.choice(universities),
        )


def fact_generator(
    num_facts: int,
    path: str = "data",
    seed: int = 0,
):
    fields = {"birthplace": [], "birthdate": [], "employer": [], "university": []}
    # Templates for facts
    for field in fields:
        with open(path + "/templates/" + field + ".txt", "r") as f:
            fields[field] = [line.strip() for line in f]

    for profile in generate_profiles(seed, path):
        kwargs = asdict(profile)
        kwargs["birthdate"] = profile.birthdate.strftime("%d %B %Y")

        for field, templates in fields.items():
            for i, template in enumerate(templates):
                yield {
                    "answer": kwargs[field],
                    "fact": template.format(**kwargs),
                    "field": field,
                    "identifier": profile.identifier,
                    "question": QUESTION_TEMPLATES[field].format(**kwargs),
                    "template": i,
                }

            # Limit the number of facts generated
            num_facts -= 1
            if num_facts <= 0:
                return
