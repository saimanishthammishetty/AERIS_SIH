import requests

def test_endpoints():
    print("Testing http://127.0.0.1:8000/health ...")
    r_health = requests.get("http://127.0.0.1:8000/health")
    print("Health:", r_health.status_code, r_health.json())

    print("Testing http://127.0.0.1:8000/ ...")
    r_root = requests.get("http://127.0.0.1:8000/")
    print("Root status:", r_root.status_code, "HTML title in page:", "AERIS" in r_root.text)

    print("Testing http://127.0.0.1:8000/predict ...")
    feats = {
        "RMS": 0.05,
        "ZeroCrossingRate": 0.08,
        "SpectralCentroid": 550.0,
        "SpectralSpread": 350.0,
        "SpectralRolloff": 1100.0,
        "SpectralFlux": 0.015,
        "Energy_0_500": 0.45,
        "Energy_500_1000": 0.30,
        "Energy_1000_2000": 0.15,
        "Energy_2000_4000": 0.07,
        "Energy_4000_8000": 0.03,
        "SpectralEntropy": 0.78,
        "SpectralFlatness": 0.03,
        "SpectralCrest": 14.5,
        "SpectralSlope": -0.0012,
        "SpectralKurtosis": 3.4,
        "SpectralSkewness": 1.2,
        "HarmonicRatio": 0.65,
        "Pitch": 125.0,
        "MFCC_1": -180.0,
        "MFCC_2": 95.0,
        "MFCC_3": -18.0,
        "MFCC_4": 12.0,
        "MFCC_5": -4.0,
        "MFCC_6": 3.0,
        "MFCC_7": -2.0,
        "MFCC_8": 0.6,
        "MFCC_9": -0.3,
        "MFCC_10": 0.2,
        "MFCC_11": -0.06,
        "MFCC_12": 0.03,
        "MFCC_13": -0.01
    }
    r_pred = requests.post("http://127.0.0.1:8000/predict", json={"features": feats})
    print("Predict response:", r_pred.status_code, r_pred.json())

    print("Testing http://127.0.0.1:8000/diagnostics ...")
    r_diag = requests.get("http://127.0.0.1:8000/diagnostics")
    print("Diagnostics:", r_diag.status_code, r_diag.json()["overall_status"])

if __name__ == "__main__":
    test_endpoints()
