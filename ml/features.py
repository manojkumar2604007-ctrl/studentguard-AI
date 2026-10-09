"""Form field definitions for the assessment page (dataset column names kept)."""
COURSES = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication", 9853: "Basic Education",
    9991: "Management (evening)",
}
YESNO = {0: "No", 1: "Yes"}

# (column, label, kind, extra)
KEY_FIELDS = [
    ("Course", "Course", "select", COURSES),
    ("Age at enrollment", "Age at enrollment", "number", dict(min=15, max=80, step=1)),
    ("Gender", "Gender", "select", {0: "Female", 1: "Male"}),
    ("Previous qualification (grade)", "Previous qualification grade (0-200)", "number", dict(min=0, max=200, step=0.1)),
    ("Admission grade", "Admission grade (0-200)", "number", dict(min=0, max=200, step=0.1)),
    ("Scholarship holder", "Scholarship holder", "select", YESNO),
    ("Debtor", "Debtor", "select", YESNO),
    ("Tuition fees up to date", "Tuition fees up to date", "select", YESNO),
    ("Displaced", "Displaced", "select", YESNO),
    ("International", "International student", "select", YESNO),
    ("Curricular units 1st sem (enrolled)", "1st sem - units enrolled", "number", dict(min=0, max=30, step=1)),
    ("Curricular units 1st sem (evaluations)", "1st sem - evaluations", "number", dict(min=0, max=50, step=1)),
    ("Curricular units 1st sem (approved)", "1st sem - units approved", "number", dict(min=0, max=30, step=1)),
    ("Curricular units 1st sem (grade)", "1st sem - average grade (0-20)", "number", dict(min=0, max=20, step=0.01)),
    ("Curricular units 2nd sem (enrolled)", "2nd sem - units enrolled", "number", dict(min=0, max=30, step=1)),
    ("Curricular units 2nd sem (evaluations)", "2nd sem - evaluations", "number", dict(min=0, max=50, step=1)),
    ("Curricular units 2nd sem (approved)", "2nd sem - units approved", "number", dict(min=0, max=30, step=1)),
    ("Curricular units 2nd sem (grade)", "2nd sem - average grade (0-20)", "number", dict(min=0, max=20, step=0.01)),
]
KEY_NAMES = [k[0] for k in KEY_FIELDS]