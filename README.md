# Indian Currency Authentication System

This is a Streamlit app that demonstrates a simple Indian currency (rupee) note authentication prototype using synthetic training data and basic image analysis techniques (edge detection, frequency analysis, micro-lettering proxies, etc.). It's intended for educational purposes only and is not an official tool for authenticating currency.

**Files included**
- [Untitled-1.py](Untitled-1.py) (original app provided)
- [app.py](app.py) (copy of the app, recommended entrypoint for Streamlit)
- [requirements.txt](requirements.txt)
- [run_app.bat](run_app.bat) (Windows convenience script)

**Requirements**
Install required Python packages (recommended inside a virtual environment):

```
python -m pip install -r requirements.txt
```

**Run the app**
Start the Streamlit app from the workspace root:

```
streamlit run app.py
```

Or on Windows double-click `run_app.bat` (or run it from PowerShell/Command Prompt):

```
run_app.bat
```

**Notes & tips**
- Use a recent Python 3.10+ interpreter for best compatibility.
- If you prefer the original filename, you can also run:

```
streamlit run "Untitled-1.py"
```

- The app performs heuristic image analysis and uses a synthetic dataset; results are illustrative and for learning only.

**License & Disclaimer**
For educational use only. This is not an official Reserve Bank of India tool.

---


Styling and assets
------------------

This project includes simple visual branding and styling to make the demo app look more attractive:

- `assets/logo.svg` — a small SVG logo displayed in the header.
- `assets/style.css` — custom CSS injected into Streamlit to provide a darker, modern theme and improved card visuals.

If you want further polish (custom fonts, screenshots in README, or alternative color palettes), tell me which direction you prefer and I will update the assets.
