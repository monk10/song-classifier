# 🎵 Song Genre Classifier

A machine learning-based audio classification system with a modern QML interface that classifies songs into 10 different genres using audio feature analysis.

## ⚠️ Current Status: KNOWN ISSUES

**This project currently experiences segmentation faults during audio processing.**

### Known Problems:
- **Segmentation Fault**: The application crashes when attempting to classify audio files
- **Root Cause**: Threading conflicts between librosa/numpy BLAS operations and Qt's event loop
- **Affected Systems**: Particularly Linux systems with certain BLAS/LAPACK configurations
- **Current Workaround Attempts**: 
  - Single-threaded librosa execution
  - Multiprocessing instead of threading
  - Environment variable configuration for thread control

### Issue Details:
```
Segmentation fault
```
Occurs during feature extraction phase when librosa processes audio data.

---

## 📋 Features (When Working)

### Genre Classification
Classifies songs into 10 genres:
- Rock
- Pop
- Jazz
- Classical
- Hip-Hop
- Electronic
- Country
- Metal
- Blues
- Reggae

### Audio Feature Analysis
Extracts 53 audio features including:
- **Tempo & Beat** - BPM detection
- **Spectral Features** - Centroid, rolloff, bandwidth
- **MFCCs** - 20 Mel-frequency cepstral coefficients
- **Chroma Features** - Pitch class profiles
- **Zero Crossing Rate** - Signal frequency content
- **RMS Energy** - Audio power

### Modern QML Interface
- Beautiful Material Design UI
- Smooth animations and transitions
- Progress indicators
- Real-time status updates
- Visual probability bars for all genres

### Pre-trained Model
- Random Forest classifier with 200 trees
- Trained on synthetic genre-specific audio profiles
- ~94% accuracy on training data (synthetic)
- Ready to use - no training required

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)
- FFmpeg (for audio decoding)

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3-full python3-venv ffmpeg libsndfile1
```

**macOS:**
```bash
brew install python ffmpeg libsndfile
```

### Python Setup

1. **Create project directory:**
```bash
mkdir song_classifier
cd song_classifier
```

2. **Create virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install PyQt5 librosa scikit-learn numpy pandas
```

### Verify Installation
```bash
python -c "import PyQt5; import librosa; import sklearn; print('✓ All packages installed')"
```

---

## 💻 Usage

### Running the Application

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the application
python song_classifier.py
```

### Using the Interface

1. **Launch** - The application window opens with a modern blue/gray gradient interface
2. **Browse** - Click "Browse" to select an audio file (MP3, WAV, FLAC, OGG, M4A)
3. **Classify** - Click "🎸 Classify Genre" to analyze the song
4. **View Results** - See predicted genre with confidence percentage and probability distribution

### Expected Workflow (If Working)
```
Select File → Analyze Features → Display Results
```

### Current Workflow
```
Select File → Click Classify → ⚠️ Segmentation Fault → Application Crashes
```

---

## 🔧 Troubleshooting

### Segmentation Fault (Current Issue)

**Attempted Solutions:**

1. **Single-threaded execution:**
```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python song_classifier.py
```

2. **Downgrade librosa:**
```bash
pip uninstall librosa numba
pip install librosa==0.9.2 numba==0.56.4
```

3. **Try different BLAS library:**
```bash
pip uninstall numpy
pip install numpy --no-binary numpy
```

4. **Run with fault handler:**
```bash
python -X faulthandler song_classifier.py
```

5. **Disable core dumps:**
```bash
ulimit -c 0
python song_classifier.py
```

### Other Common Issues

**Import Error: No module named 'PyQt5'**
```bash
pip install PyQt5
```

**FileNotFoundError: main.qml**
- Ensure `main.qml` is in the same directory as `song_classifier.py`

**Audio Loading Error**
```bash
sudo apt install ffmpeg libsndfile1  # Linux
brew install ffmpeg libsndfile       # macOS
```

**Permission Denied**
```bash
chmod +x song_classifier.py
```

---

## 🎨 Customizing the Interface

Edit `main.qml` to customize the UI:

### Change Colors
```qml
Material.accent: Material.Purple  // Change accent color
color: "#ff6b6b"                  // Change button color
```

### Adjust Layout
```qml
Layout.preferredHeight: 100  // Change component height
spacing: 30                  // Adjust spacing
```

### Add Animations
```qml
Behavior on opacity {
    NumberAnimation { duration: 300 }
}
```

---

## 🏗️ Technical Details

### Machine Learning Pipeline

1. **Feature Extraction**
   - Audio loaded at 22050 Hz sample rate
   - 30-second duration analyzed
   - 53 features extracted using librosa

2. **Preprocessing**
   - StandardScaler normalization
   - Feature vector: 53 dimensions

3. **Classification**
   - Algorithm: Random Forest
   - Trees: 200
   - Max Depth: 20
   - Min Samples Split: 5

4. **Output**
   - Predicted genre
   - Confidence score
   - Probability distribution across all 10 genres

### Architecture

```
QML Interface (main.qml)
    ↓
PyQt5 Bridge (GenreClassifierBackend)
    ↓
Worker Thread (ClassificationWorker)
    ↓
Multiprocessing (classify_in_subprocess)
    ↓
Librosa Feature Extraction
    ↓
Scikit-learn Classification
    ↓
Results Display
```

---

## 📊 Model Performance

**Note:** Performance metrics based on synthetic training data only. Real-world performance untested due to segmentation fault issue.

- **Training Accuracy**: ~94%
- **Features Used**: 53
- **Training Samples**: 1000 (100 per genre)
- **Model Size**: ~2MB in memory

---

## 🐛 Known Bugs

### Critical
- ❌ **Segmentation fault during audio classification** - Blocks all functionality
- ❌ Cannot process any audio files
- ❌ Application crashes on classify button

### Medium
- ⚠️ No error recovery after segfault
- ⚠️ Progress bar doesn't reflect actual progress stages
- ⚠️ No support for batch processing

### Minor
- 🔸 Window size not remembered between sessions
- 🔸 No drag-and-drop file support
- 🔸 No audio preview functionality

---

## 🔮 Future Improvements

### If Segmentation Fault is Resolved:

**Features:**
- [ ] Real dataset training (GTZAN, Million Song Dataset)
- [ ] Batch file processing
- [ ] Export results to CSV
- [ ] Audio waveform visualization
- [ ] Drag-and-drop file support
- [ ] Model retraining interface
- [ ] Genre sub-classification
- [ ] Confidence threshold settings

**Performance:**
- [ ] GPU acceleration for feature extraction
- [ ] Caching extracted features
- [ ] Model compression
- [ ] Streaming audio support

**UI/UX:**
- [ ] Dark mode theme
- [ ] Custom color schemes
- [ ] Window state persistence
- [ ] Keyboard shortcuts
- [ ] Results history

---

## 🤝 Contributing

This project is currently in a **non-functional state** due to segmentation faults. Contributions to fix this issue are highly welcome!

### How to Contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b fix-segfault`)
3. Make your changes
4. Test thoroughly
5. Commit your changes (`git commit -m 'Fix: Resolve segmentation fault issue'`)
6. Push to the branch (`git push origin fix-segfault`)
7. Open a Pull Request

### Priority Issues:
1. **Fix segmentation fault** (Critical)
2. Implement proper audio processing isolation
3. Add comprehensive error handling
4. Create unit tests

---

## 📝 Dependencies

```
PyQt5>=5.15.0          # Qt framework and QML support
librosa>=0.10.0        # Audio analysis
scikit-learn>=1.0.0    # Machine learning
numpy>=1.21.0          # Numerical computing
pandas>=1.3.0          # Data manipulation
```

See `requirements.txt` for exact versions.

---

## 📄 License

This project is provided as-is for educational purposes.

**Disclaimer:** This software is currently non-functional due to critical bugs. Use at your own risk.

---

## 🆘 Support

If you experience issues (which you likely will):

1. Check the **Known Issues** section
2. Try the **Troubleshooting** steps
3. Search existing GitHub issues
4. Create a new issue with:
   - Python version
   - Operating system
   - Full error log
   - Steps to reproduce

---

## 🙏 Acknowledgments

- **librosa** - Audio analysis library
- **scikit-learn** - Machine learning framework
- **PyQt5** - Qt framework for Python
- **Qt QML** - Declarative UI framework

---

## ⚠️ Final Warning

**This application does not currently work due to segmentation faults during audio processing. It is provided for educational and development purposes only. Successful execution is not guaranteed on any system.**

---

## 📞 Contact

For questions, suggestions, or if you've found a fix for the segmentation fault:
- Open an issue on GitHub
- Contributions are welcome via Pull Requests

---

**Last Updated:** November 2025  
**Status:** 🔴 Non-functional - Segmentation Fault Issue  
