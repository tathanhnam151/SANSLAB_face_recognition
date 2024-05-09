import requests, threading, json

def feature_backup():
    def send_request(payload):
        url = "https://script.google.com/macros/s/AKfycbxz9msoEMbfCL2ffGjoVAmGAXgK9FmFj067zqAoieZRT_hBEMM6luFcPLOp2XlCXYEucg/exec"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        return response

    # Load the JSON data from a file
    with open('database/photo_datasets/face_features/feature.json') as f:
        data = json.load(f)

    # Start a new thread to send the first item in the JSON array
    threading.Thread(target=send_request, args=(data[0],)).start()

def record_student_attend(mssv):
    def send_request(mssv):
        url = f"https://script.google.com/macros/s/AKfycbxz9msoEMbfCL2ffGjoVAmGAXgK9FmFj067zqAoieZRT_hBEMM6luFcPLOp2XlCXYEucg/exec?mssv={mssv}"
        print(url)
        response = requests.post(url)    
        return response
    threading.Thread(target=send_request, args=(mssv,)).start()
        