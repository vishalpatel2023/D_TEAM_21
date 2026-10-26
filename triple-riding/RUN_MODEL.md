# Run the Triple-Riding Detection Model

This guide explains how to run the project on your local machine and save the output image or JSON result.

## 1) Open the project folder (use powershell)

```powershell
cd "C:\Users\sojwa\OneDrive\Desktop\IPCV proj\road-safety-monitoring-system\triple-riding"
```

## 2) Use the project's Python environment

To avoid using a different system Python that may not have the required
packages, the run commands below invoke `.venv\Scripts\python.exe` directly.
This works in both PowerShell and Command Prompt; you do not need to activate
the environment first.

If you prefer to activate it, use the command for your terminal:

```powershell
# PowerShell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# Command Prompt (cmd.exe)
.\.venv\Scripts\activate.bat
```

## 3) Run the model on the sample images

```powershell
.\.venv\Scripts\python.exe run.py --input inputs\single.png --output outputs\single_result.jpg
.\.venv\Scripts\python.exe run.py --input inputs\double.png --output outputs\double_result.jpg
.\.venv\Scripts\python.exe run.py --input inputs\triple.png --output outputs\triple_result.jpg
.\.venv\Scripts\python.exe run.py --input inputs\triple2.png --output outputs\triple2_result.jpg
```

Each command will:

- detect people and motorcycles,
- count riders per motorcycle,
- identify triple-riding violations,
- save the annotated output image in the `outputs` folder.

Tested sample results:

| Input                | Riders detected | Violations | Output                       |
| -------------------- | --------------: | ---------: | ---------------------------- |
| `inputs\single.png`  |               1 |          0 | `outputs\single_result.jpg`  |
| `inputs\double.png`  |               2 |          0 | `outputs\double_result.jpg`  |
| `inputs\triple.png`  |               3 |          1 | `outputs\triple_result.jpg`  |
| `inputs\triple2.png` |               3 |          1 | `outputs\triple2_result.jpg` |

For a different image, replace the input path with its filename in `inputs`
and choose an output filename in `outputs`. Do not type a placeholder such as
`your_image.png` literally; it must be replaced with the name of an actual
file.

## 4) Run the model and print JSON output

```powershell
.\.venv\Scripts\python.exe run.py --input inputs\double.png --output outputs\double_result.jpg --json
```

This prints structured JSON containing counts and detection details in the terminal.

## 5) Run on another image

```powershell
.\.venv\Scripts\python.exe run.py --input inputs\triple2.png --output outputs\triple2_result.jpg
```

## 6) Run on a video

```powershell
.\.venv\Scripts\python.exe run.py --input inputs\traffic.mp4 --output outputs\result.mp4
```

## 7) Override confidence threshold

```powershell
.\.venv\Scripts\python.exe run.py --input inputs\triple2.png --output outputs\triple2_result.jpg --confidence 0.4
```

## 8) Output folder

All processed images/videos are saved to:

```text
C:\Users\sojwa\OneDrive\Desktop\IPCV proj\road-safety-monitoring-system\triple-riding\outputs
```

## 9) Example result

After running successfully, the script prints a summary such as:

```json
{
  "module": "triple_riding",
  "persons_detected": 4,
  "motorcycles_detected": 2,
  "violations": 1
}
```

## 10) Notes

- The model downloads `yolov8n.pt` automatically the first time it runs.
- Supported input image formats include `.png`, `.jpg`, `.jpeg`, `.bmp`.
- Supported video formats include `.mp4`, `.avi`, `.mov`, `.mkv`.

If you want, I can also create a short `HOW_TO_RUN.md` version with exact commands for Windows and Linux/macOS side by side.
