from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
import pytesseract
import numpy as np
import cv2
import io
import re
from rapidfuzz import fuzz

# Configure Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize FastAPI application
app = FastAPI(title="Lab Report OCR Service")

# === Pydantic models ===
class TestResult(BaseModel):
    test_name: str
    test_value: str
    test_unit: str
    bio_reference_range: str
    lab_test_out_of_range: bool

class OCRResult(BaseModel):
    is_success: bool
    data: List[TestResult]

# === Image preprocessing ===
def prepare_image(frame: np.ndarray) -> np.ndarray:
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    binary = cv2.adaptiveThreshold(
        scaled, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        35, 10
    )
    return binary

# === Reference list for lab tests ===
LAB_TESTS_REFERENCE = [
    "HB ESTIMATION", "HAEMOGLOBIN", "PCV", "KETONE BODIES", "URINE FOR KETONES",
    "BLOOD SUGAR", "RBC COUNT", "WBC COUNT", "PLATELET COUNT",
    "NEUTROPHILS", "LYMPHOCYTES", "TRIGLYCERIDES", "CHOLESTEROL",
    "BILIRUBIN", "CREATININE", "UREA", "SODIUM", "POTASSIUM"
    # Blood parameters
            "hemoglobin", "hb", "rbc", "wbc", "platelet", "neutrophil", "lymphocyte", 
            "monocyte", "eosinophil", "basophil", "hematocrit", "mcv", "mch", "mchc",
            "rdw", "esr", "pcv", "tlc",
            
            # Liver function parameters
            "bilirubin", "sgpt", "sgot", "alkaline phosphatase", "total protein", 
            "albumin", "globulin", "a/g ratio", "gamma glutamyl", "transferase",
            
            # Kidney function parameters
            "urea", "creatinine", "uric acid", "bun", "gfr",
            
            # Urine examination parameters
            "color", "transparency", "specific gravity", "ph", "albumin", "sugar", 
            "ketone", "urobilinogen", "pus cells", "rbc", "epithelial cells", 
            "cast", "crystal", "bacteria", "yeast cells",
            
            # Lipid profile
            "cholesterol", "triglyceride", "hdl", "ldl", "vldl",
            
            # Other common tests
            "glucose", "sodium", "potassium", "chloride", "calcium", "phosphorus",
            "magnesium", "tsh", "t3", "t4", "hba1c"
]

def is_known_test(label: str) -> bool:
    return any(keyword in label for keyword in LAB_TESTS_REFERENCE)

def get_closest_test(label: str) -> str:
    best_match = max(LAB_TESTS_REFERENCE, key=lambda ref: fuzz.ratio(ref, label))
    return best_match if fuzz.ratio(best_match, label) > 75 else label

# === OCR text parsing ===
def parse_ocr_text(ocr_text: str) -> List[TestResult]:
    results = []
    lab_pattern = re.compile(r"""
        (?P<name>[A-Z\s\(\)\-/]+?)[:\-]?\s+         # Test name
        (?P<value>[A-Z]+|\d+\.?\d*)\s*              # Value (e.g. NEGATIVE or 9.4)
        (?P<unit>[a-zA-Z/%µ\d])?\s               # Unit (optional)
        \(?(?P<range>\d+\.?\d*\s*-\s*\d+\.?\d*)?\)? # Ref range (optional)
    """, re.VERBOSE | re.IGNORECASE)

    for m in lab_pattern.finditer(ocr_text):
        raw_label = m.group("name").strip().upper()
        val_str = m.group("value").strip().upper()
        unit_label = (m.group("unit") or "").strip()
        ref_range = (m.group("range") or "").strip().replace(" ", "")

        matched_name = get_closest_test(raw_label)
        if not is_known_test(matched_name) and len(matched_name) < 4:
            continue

        out_of_range_flag = False
        try:
            if val_str.replace('.', '', 1).isdigit() and ref_range:
                low, high = map(float, ref_range.split("-"))
                numeric_val = float(val_str)
                out_of_range_flag = not (low <= numeric_val <= high)
        except:
            pass

        results.append(TestResult(
            test_name=matched_name,
            test_value=val_str,
            test_unit=unit_label,
            bio_reference_range=ref_range,
            lab_test_out_of_range=out_of_range_flag
        ))
    return results

# === API endpoint ===
@app.post("/get-lab-tests", response_model=OCRResult)
async def process_lab_report(report: UploadFile = File(...)):
    try:
        file_bytes = await report.read()
        image_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # Perform OCR
        binary_img = prepare_image(image_cv)
        ocr_text = pytesseract.image_to_string(binary_img, config="--psm 4")

        # Extract test results
        parsed_tests = parse_ocr_text(ocr_text)

        return OCRResult(is_success=True, data=parsed_tests)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
