from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
D=Document()
for s in D.sections: s.left_margin=s.right_margin=Inches(1); s.top_margin=s.bottom_margin=Inches(0.9)
st=D.styles['Normal']; st.font.name='Calibri'; st.font.size=Pt(10.5)
for n,sz in (('Heading 1',18),('Heading 2',14),('Heading 3',11.5)):
    h=D.styles[n]; h.font.name='Calibri'; h.font.size=Pt(sz); h.font.color.rgb=RGBColor(0x14,0x4a,0x6b)
def P(t,b=False,i=False):
    p=D.add_paragraph(); r=p.add_run(t); r.bold=b; r.italic=i; p.paragraph_format.space_after=Pt(5); return p
def B(items):
    for t in items: D.add_paragraph(t,style='List Bullet')
def N(items):
    for t in items: D.add_paragraph(t,style='List Number')
def shade(c,color):
    tc=c._tc.get_or_add_tcPr(); s=OxmlElement('w:shd'); s.set(qn('w:val'),'clear'); s.set(qn('w:fill'),color); tc.append(s)
def T(rows,widths=None):
    t=D.add_table(rows=len(rows),cols=len(rows[0])); t.style='Table Grid'
    for i,r in enumerate(rows):
        for j,v in enumerate(r):
            c=t.cell(i,j); c.text=''; run=c.paragraphs[0].add_run(str(v)); run.font.size=Pt(9.5)
            if i==0: run.bold=True; shade(c,'DCE9F2')
    if widths:
        for r in t.rows:
            for j,w in enumerate(widths): r.cells[j].width=Inches(w)
    D.add_paragraph()
def IMG(path,w=6.2,cap=None):
    D.add_picture(path,width=Inches(w)); D.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
    if cap: p=P(cap,i=True); p.alignment=WD_ALIGN_PARAGRAPH.CENTER

t=D.add_heading('MotionProof',0)
P('Interview manual: how the project works, why each choice was made, what the evidence shows and where it fails',b=True)
P('Entry for AI OpenBuild (Devpost), November 2026. Entrant: M J Anthony. Built with AI assistance (see section 12).')
P('This manual lets you explain every part of the project in an interview without having built it by hand. Numbers are copied from the measured result files in the repository. Where a figure depends on how it was measured, the measurement is named next to it.',i=True)

D.add_heading('1. The project in one minute',1)
P('MotionProof is a browser app that answers one question about wearable activity recognition: how well does a model work on a person it has never seen, and how does it fail when the data goes wrong? Three models were trained for this (feature logistic regression, a small 1D CNN, and the same CNN trained with augmentation) on the HARTH free-living accelerometer dataset. The app runs them live in the browser with no server, shows truth next to every model, reports subject-held-out evidence, lets the user damage the data in a stress lab, and accepts the user\'s own CSV.')
P('The main finding is deliberately not "the CNN wins". Augmentation helps robustness a lot. A plain CNN is no better than a good feature model when judged on pooled cross-validation. All models collapse when the thigh sensor is lost, and they stay confident while doing so.')
D.add_heading('Key facts to memorise',3)
T([['Item','Value'],
['Data','HARTH, UCI #779, CC BY 4.0. 22 adults, about 35.9 hours, back + right thigh accelerometers, 50 Hz, free living, camera-annotated'],
['Task','9 classes: walking, running, shuffling, stairs up, stairs down, standing, sitting, lying, cycling (four cycling codes merged)'],
['Windows','2 s (100 samples, 6 channels), 50% overlap, mixed-label windows dropped: 118,495 windows'],
['Models','Feature LR (67 features), 1D CNN (about 17k parameters), same CNN with augmentation'],
['Evaluation','5-fold CV with whole subjects held out, plus a showcase split of 4 held-out subjects (6, 13, 20, 28) with 3 seeds'],
['Best result','CNN+aug: pooled CV macro-F1 0.853, accuracy 0.954'],
['Runs in','One offline HTML file (about 1.2 MB), about 1.2 ms per window per model in the browser'],
['Tests','28 parity and validation checks (JS vs PyTorch), 8 headless-browser UI checks']],[1.4,5.1])

D.add_heading('2. Problem framing and originality',1)
P('Most activity-recognition demos show one accuracy number from a random split of windows. Windows from the same person are near-duplicates, so a random split leaks identity and inflates scores. The honest question is performance on new people. Wearables also fail in ordinary ways (a sensor falls off, is worn rotated, is noisy), and a model that stays confident when it is broken is dangerous in any product built on it.')
P('What is different here: (1) every number is subject-held-out; (2) the app makes the failure visible to a non-expert by letting them damage the data live; (3) confidence calibration and abstention are measured, not assumed; (4) inference is re-implemented in JavaScript from exported weights and checked against PyTorch, so the demo is the real model and not a mock-up.')

D.add_heading('3. Screen-by-screen walkthrough',1)
P('The app has five tabs. Screenshots are from the built app.')
D.add_heading('3.1 Live bench',2)
IMG('screenshots/bench.png',6.3,'Live bench: raw signals, truth strip and three model strips.')
B(['Shows a real held-out excerpt (from subjects the models never trained on). Top: the six channels. Below: the true activity strip and one strip per model. Wrong windows are marked.',
'Abstain slider: below the chosen confidence the model says "unsure" instead of guessing. The panel reports coverage (share of windows answered) and accuracy on the answered ones. Raising the threshold trades coverage for accuracy.',
'Smoothing: a majority vote over neighbouring windows. It removes single-window flicker but delays transitions. It is optional and off by default so the raw model is what you see.',
'Why it matters: a user can see where and when models disagree with the truth, not only a score.'])
IMG('screenshots/bench_thr.png',6.0,'Abstain threshold raised: fewer windows answered, higher accuracy on those.')
D.add_heading('3.2 Evidence',2)
IMG('screenshots/evidence.png',6.3,'Evidence tab: every figure labelled by model and by how it was measured.')
B(['Table of macro-F1, accuracy and calibration error for the three models. Pooled cross-validation, mean of folds with spread, and the 4-subject showcase split (mean of 3 seeds) are separate, labelled columns.',
'Per-class F1 and confusion matrices: shows shuffling and the two stair classes are weak.',
'Per-subject F1: ranges from 0.48 to 0.96 across people, so the average hides a bad worst case.',
'Reliability (calibration) view: how closely confidence tracks accuracy. Temperature scaling was fitted on validation subjects only.'])
D.add_heading('3.3 Stress lab',2)
IMG('screenshots/stress.png',6.3,'Stress lab: damage the signal and watch predictions change.')
B(['Controls: pick a damage type (for example noise, gain, sensor rotation, time stretch, clipping, lose the back sensor, lose the thigh sensor) and its strength.',
'The app re-runs all three models in the browser on the damaged signal and shows accuracy before and after, how many predictions changed and mean confidence.',
'The offline-measured version of the same damage over all held-out windows is in the table in section 7.'])
D.add_heading('3.4 Import',2)
IMG('screenshots/import.png',6.0,'Import: validates a CSV before running models.')
B(['Accepts a CSV of six accelerometer columns (back x,y,z then thigh x,y,z, optional timestamp). The validator refuses with a reason when the file is empty, columns are missing or there are fewer than 100 usable rows. It warns and fixes what it can: skips rows with missing or non-numeric values, estimates the sampling rate from timestamps and resamples to 50 Hz when it is more than 10 percent off, assumes 50 Hz when there is no timestamp, and converts m/s squared to g when magnitudes are far above 1.',
'Data never leaves the browser. A sample file is provided (sample_import.csv).',
'Limit: the models assume the HARTH sensor placement (lower back and right thigh) and orientation. Other placements will give meaningless output; the app says so.'])
IMG('screenshots/imported.png',6.0,'An imported recording after validation and prediction.')
D.add_heading('3.5 History and model card',2)
IMG('screenshots/history.png',6.0,'Saved sessions and the model card.')
B(['Sessions are saved in the browser (local storage) so runs can be compared. The experiment record can be exported as JSON.',
'Model card: intended use, data, metrics with their protocol, limits, and a "not a medical device" statement.'])

D.add_heading('4. Data and preprocessing',1)
B(['Why HARTH: free-living data (people did what they normally do, not a scripted lab routine), two sensors, 22 people, and a clear CC BY 4.0 licence on its UCI page. UCI HAR was considered first but its bundled README and its UCI page disagree on commercial use, so it was dropped to avoid a licensing risk.',
'Cycling labels (sitting, standing, walking, running and cycling variants) were merged: the four cycling codes became one class. 12 original codes became 9 classes.',
'Windowing: 2 seconds at 50 Hz = 100 samples, 6 channels (back x/y/z, thigh x/y/z). 50% overlap. A window is kept only if every sample has the same label (pure windows), so transitions are excluded. This gives cleaner labels but means transitions are not evaluated.',
'Normalisation: per-channel mean and standard deviation from training subjects only, applied to everything else.',
'Gaps: the recordings have timestamp gaps. Windows never span a gap.',
'Class balance: sitting is the largest class (57,854 of 118,495 windows); shuffling has 1,781 windows and the two stair classes 1,096 and 838. Rare classes are the weak ones.'])
P('Window counts by class: walking 21,233; running 5,675; shuffling 1,781; stairs up 1,096; stairs down 838; standing 11,654; sitting 57,854; lying 8,431; cycling 9,933.',i=True)

D.add_heading('5. Models, math and algorithms',1)
D.add_heading('5.1 Feature logistic regression',2)
P('Each window becomes 67 numbers: per-channel mean, standard deviation, minimum, maximum and jerk (mean absolute first difference), mean and spread of each sensor acceleration magnitude, the share of signal energy in several frequency bands (fixed DFT basis), and correlations between matching back and thigh axes. A softmax regression maps features z to class probabilities:')
P('p(k | x) = exp(w_k . z + b_k) / sum_j exp(w_j . z + b_j)',b=True)
P('Trained with Adam (lr 0.01, weight decay 1e-4), cross-entropy with class weights proportional to 1/sqrt(class frequency). Why keep it: it is the baseline that proves the deep model is worth using. It is also calibrated well (ECE 0.003).')
D.add_heading('5.2 1D CNN',2)
T([['Layer','Shape','Parameters'],
['Conv1d 6 to 24, kernel 5, BatchNorm, ReLU','100 to 24 channels','744 + BN'],
['Conv1d 24 to 48, kernel 5, BatchNorm, ReLU','48 channels','5,808 + BN'],
['Conv1d 48 to 64, kernel 3, BatchNorm, ReLU','64 channels','9,280 + BN'],
['Global average pooling over time','64','0'],
['Linear 64 to 9, softmax','9 classes','585']],[3.4,1.8,1.3])
P('Convolutions slide learned filters along time so the same movement pattern is recognised wherever it occurs in the window. Global average pooling removes the dependence on position. About 17k parameters, so it is small enough to ship in the browser and to run in about 1 ms. Trained with AdamW (lr 3e-3, weight decay 1e-3), OneCycle learning-rate schedule (max 4e-3), batch size 256, 14 epochs, class-weighted cross-entropy, early model selection on the validation subjects.')
D.add_heading('5.3 CNN with augmentation',2)
P('Same network, but each training batch is randomly perturbed: small random sensor rotations (about 12 degrees standard deviation), gain changes of plus or minus 10 percent, Gaussian noise, and dropout of a whole sensor (6 percent of windows, either sensor). The idea is to teach the model that these changes do not change the activity, which is exactly what real devices do. This is the only difference from model 2, so differences between them are caused by augmentation (within seed noise).')
D.add_heading('5.4 Calibration and abstention',2)
P('Raw network confidences are usually too high. Temperature scaling divides the logits by one number T (found by minimising the validation negative log-likelihood) and changes no prediction, only the confidence: p = softmax(z / T). Expected calibration error (ECE) bins predictions by confidence and averages |accuracy - confidence| weighted by bin size. Abstention answers only if the top probability is above a threshold.')
D.add_heading('5.5 Metrics',2)
B(['Accuracy: share correct. Dominated by big classes (sitting), so it flatters the models.',
'Macro-F1: the unweighted mean over classes of F1 = 2PR/(P+R). Each class counts equally, so weak rare classes show. This is the headline metric.',
'Pooled vs mean-of-folds: pooled puts all out-of-fold predictions together and computes one score. Mean-of-folds computes a score per fold and averages. They differ when folds differ in size or class mix. Both are reported.'])
D.add_heading('5.6 In-browser inference',2)
P('The trained weights are exported to JSON and embedded in the HTML. The JavaScript re-implements the feature extractor, the convolution, batch normalisation (folded into the running statistics), global pooling and softmax. 28 tests compare its outputs with PyTorch on stored golden vectors (CNN to float precision; LR with a 5e-3 tolerance because the Python features were computed in float32).')

D.add_heading('6. Evaluation design',1)
B(['Subject-grouped 5-fold CV: each fold holds out whole people. Inside a fold, 3 other subjects are used for early stopping and temperature fitting and the remaining ones train. Training used every second window (non-overlapping 2 s windows) for speed on CPU.',
'Showcase split used by the live app: test subjects 6, 13, 20, 28; validation subjects 9, 15, 24; the other 15 train. Three seeds. This is the split whose excerpts appear on the Live bench, so the demo shows data the models never saw.',
'Why two protocols: CV gives a number over all 22 people. The showcase split gives a fixed set of unseen people for the live demo. Their scores differ and should not be compared as if they were the same measurement.'])

D.add_heading('7. Results (measured)',1)
T([['Model','Pooled CV macro-F1','Mean of 5 folds (sd)','Pooled CV accuracy','Pooled ECE','4 held-out subjects, macro-F1, mean of 3 seeds (sd)'],
['Feature logistic regression','0.810','0.810 (0.035)','0.940','0.003','0.830 (0.009)'],
['1D CNN','0.792','0.815 (0.044)','0.928','0.031','0.865 (0.012)'],
['1D CNN + augmentation','0.853','0.855 (0.034)','0.954','0.011','0.889 (0.014)']],[1.4,0.9,1.0,0.9,0.7,1.6])
P('The figures 0.830, 0.865 and 0.889 are the 4-subject, 3-seed means for LR, CNN and CNN+aug. The figures 0.810, 0.792 and 0.853 are pooled cross-validation. They are different measurements.',b=True)
D.add_heading('7.1 Per-class F1 (pooled CV)',2)
T([['Class','LR','CNN','CNN+aug'],['shuffling','0.54','0.54','0.49'],['stairs up','0.65','0.59','0.86'],['stairs down','0.47','0.44','0.58']],[2,1,1,1])
P('The three weak classes. Easy classes such as sitting and running score above 0.85 for all three models.')
D.add_heading('7.2 Stress test (4 held-out subjects, macro-F1, mean of 3 seeds)',2)
T([['Condition','LR','CNN','CNN+aug'],
['Clean','0.83','0.87','0.89'],
['Thigh sensor lost','0.26','0.28','0.51'],
['Back sensor lost','0.21','0.13','0.78'],
['Thigh sensor rotated 30 degrees','0.67','0.60','0.75']],[2.6,1,1,1])
B(['Losing the thigh sensor wrecks every model. The thigh sensor carries the posture information that separates sitting, standing and lying.',
'Augmentation helps most for noise, rotation and a lost back sensor.',
'Confidence stays high under failure: with the thigh sensor lost, mean confidence is about 0.94 for LR and CNN while their accuracy is only 0.61 to 0.68; CNN+aug is less overconfident (mean confidence 0.76). Abstaining on confidence alone therefore does not detect a broken sensor.'])

D.add_heading('8. Numbered design decisions',1)
decs=[('Use real, free-living data','Lab data is easy and not representative. HARTH is free-living.'),
('Choose HARTH over UCI HAR','Clear CC BY 4.0 licence and two sensors, no licence contradiction.'),
('Hold out whole subjects','Random window splits leak identity and inflate results.'),
('Report macro-F1 as the headline','Accuracy hides the rare classes.'),
('Report both pooled and mean-of-fold numbers','They answer slightly different questions; hiding either would look like cherry-picking.'),
('Keep a feature baseline','It shows what the CNN actually adds; here the plain CNN adds nothing on pooled CV.'),
('Keep the network tiny','About 17k parameters runs in a browser in about 1 ms and trains on a CPU.'),
('Isolate augmentation as the only difference between models 2 and 3','So any gain can be attributed to it.'),
('Calibrate with temperature scaling on validation subjects','Never on test subjects, to avoid leakage.'),
('Re-implement inference in JavaScript and test against PyTorch','The demo must be the real model; parity tests prove it.'),
('Single offline HTML file','No server, no accounts, no data leaves the device. Anyone can open it.'),
('Stress lab uses real damage on real held-out data','Makes failure visible and gives a measured number for each damage.'),
('Drop mixed-label windows','Cleaner labels, at the cost of not scoring transitions (stated as a limit).'),
('Use 3 seeds on the showcase split','Shows training-seed noise; spread is reported.'),
('Validate imported CSVs strictly','A refusal with a reason is better than a confident wrong answer.'),
('State "not a medical device" and the placement limit in the app','The models only apply to healthy adults and this sensor placement.'),
('Disclose AI assistance in the repo and the entry','Required by the rules and honest.')]
for i,(a,b) in enumerate(decs,1): P(f'{i}. {a}. {b}')

D.add_heading('9. Honest limitations',1)
B(['22 healthy adults, one sensor placement and one device type. Not tested on other people, placements, devices or on older or clinical populations.',
'Differences between models are of the same size as fold-to-fold spread (sd 0.03 to 0.04). Only the CNN+aug advantage and the robustness gaps are clear.',
'Shuffling, stairs up and stairs down are weak and will be confused with walking.',
'Transitions between activities are not scored because mixed windows were dropped.',
'The four cycling codes are merged, so the model cannot say what kind of cycling.',
'Temperature scaling was fitted on only three validation subjects.',
'Stress tests are synthetic damage, not real hardware failures.',
'Training used half of the available windows and was on CPU; no hyperparameter search was run. Early stopping and calibration used validation subjects, not the test subjects.',
'Re-training will not reproduce numbers to the last digit (CPU floating point).',
'Not a medical device. It must not be used for diagnosis, fall detection, gait assessment or monitoring a person.'])

D.add_heading('10. How to run and reproduce',1)
B(['Open index.html in any modern browser. Nothing else is needed.',
'node build.mjs rebuilds the HTML. node tests/parity.test.js runs the 28 checks. node tests/browser.test.js runs the 8 UI checks (needs puppeteer-core and Chrome).',
'To retrain: download HARTH, follow the commands in the README (prep, run_jobs, analyze, stress, export, build_results).'])

D.add_heading('11. Interview questions and answers',1)
QA=[
('What does the project do?','It trains three activity-recognition models on real wearable data, tests them on people they never saw, stress-tests them, and lets anyone explore the results and run the models on their own CSV in the browser.'),
('Why not just report accuracy?','Accuracy is dominated by common classes like sitting. Macro-F1 gives each activity equal weight, so weak classes such as stairs show up.'),
('What is subject-held-out evaluation?','The people in the test set never appear in training. With a random window split, near-identical windows from the same person appear on both sides and the score is inflated.'),
('What is your best model and how good is it?','CNN with augmentation: pooled CV macro-F1 0.853 and accuracy 0.954; 0.889 (mean of 3 seeds) on the 4-subject showcase split. These are different measurements and I do not mix them.'),
('Is the CNN better than logistic regression?','Not clearly. Plain CNN pooled CV macro-F1 is 0.792 against 0.810 for LR. The mean-of-folds numbers (0.815 vs 0.810) are within noise. Augmentation is what makes the deep model clearly better.'),
('Why do the numbers 0.810 and 0.830 differ for LR?','0.810 is pooled cross-validation across all 22 people; 0.830 is the mean of three seeds on four specific held-out people. Different people, different protocol.'),
('What are pooled and mean-of-folds scores?','Pooled computes one score over all out-of-fold predictions. Mean-of-folds computes a score per fold and averages. I show both.'),
('Which classes are hard?','Shuffling (F1 0.49 to 0.54), stairs up (0.59 to 0.86) and stairs down (0.44 to 0.58). They are rare and look like walking.'),
('What happens when a sensor is lost?','Performance collapses. Without the thigh sensor macro-F1 is 0.26 (LR), 0.28 (CNN) and 0.51 (CNN+aug). Without the back sensor, LR and CNN fall to 0.21 and 0.13 but CNN+aug keeps 0.78.'),
('Why does augmentation help?','It shows the network during training the kinds of change that happen in real use: rotation, noise, speed, a missing sensor. The model learns not to rely on a single channel.'),
('What is the worrying finding?','Confidence stays high when the model is wrong. With the thigh sensor lost, mean confidence is about 0.94 for LR and CNN while accuracy is 0.61 to 0.68. So a confidence threshold alone cannot catch a broken sensor.'),
('What is calibration and ECE?','Calibration means a stated 80 percent confidence is right about 80 percent of the time. ECE is the average gap between confidence and accuracy over confidence bins. Pooled ECE: LR 0.003, CNN 0.031, CNN+aug 0.011.'),
('What is temperature scaling?','Divide the logits by one number fitted on validation data. It changes confidence but never the predicted class.'),
('Why did you choose HARTH?','Free-living data, two sensors, 22 subjects and a CC BY 4.0 licence on its UCI page. UCI HAR had conflicting licence statements.'),
('How are windows made?','2 seconds at 50 Hz, 100 samples by 6 channels, 50 percent overlap, only windows with a single label.'),
('What are the downsides of dropping mixed windows?','Cleaner labels, but transitions are never evaluated, so real continuous use will look worse near transitions.'),
('What is in the 67 features?','Per-channel mean, standard deviation, min, max and jerk, per-sensor acceleration magnitude statistics, the share of energy in frequency bands, and back-to-thigh correlations.'),
('Why a 1D convolution?','Accelerometer windows are time series; a filter slides along time and detects a movement pattern anywhere in the window with few parameters.'),
('What does batch normalisation do?','It rescales activations per channel during training to make optimisation stable; at inference it is a fixed scale and shift, folded into the layer in the browser version.'),
('How many parameters and how fast?','About 17,000. About 1.2 ms per window per model in the browser (measured in headless Chrome on a CPU-only sandbox).'),
('How do you know the browser version is the same model?','28 tests compare JavaScript outputs with PyTorch on stored golden vectors. CNN matches to float precision; LR within 5e-3 because of float32 features.'),
('What does the abstain slider do?','It refuses to answer when top confidence is below the threshold, and reports coverage and accuracy on what it did answer. It helps for ordinary errors but not for sensor failure.'),
('What is smoothing for?','A majority vote across neighbouring windows removes single-window flicker. It adds delay near transitions.'),
('What does the Import tab check?','Column count, numeric values, unit (g or m/s squared), sampling rate and length. It refuses with a reason if something is off.'),
('Does user data leave the device?','No. Everything runs locally in the browser and there is no server or model API.'),
('How did you handle class imbalance?','Class weights proportional to 1 over the square root of class frequency in the loss, and macro-F1 for evaluation.'),
('How reproducible is it?','Seeds are fixed and the code is in the repo, but CPU floating-point differences mean a rerun is not bit-exact. The shipped numbers come from stored result files.'),
('Is this a medical or safety tool?','No. It is a research and teaching bench, trained on healthy adults with one placement. It must not be used for diagnosis or monitoring a person.'),
('How was AI used?','An AI assistant wrote the code, ran training and evaluation and wrote the documentation, directed by the entrant. The README and Devpost entry say so. All numbers come from the stored scripts and outputs.'),
('What would you do next?','Add leave-one-device-out tests, real sensor-fault recordings, a model that detects a missing sensor directly instead of relying on confidence, and a larger and more diverse dataset.'),
('What is the single most important lesson?','A good headline score hides how a model fails. Test on new people, look at the worst cases, and check that confidence means something.'),
('If a judge says the plain CNN underperforms, is that a bug?','No. It is a reported finding. On pooled CV it is 0.792 against 0.810 for LR and 0.853 with augmentation. I kept it because it makes the augmentation effect attributable.'),
]
for i,(q,a) in enumerate(QA,1):
    p=D.add_paragraph(); r=p.add_run(f'Q{i}. {q}'); r.bold=True; p.paragraph_format.space_after=Pt(1); p.paragraph_format.keep_with_next=True
    P(a)

D.add_heading('12. AI-assistance disclosure and credits',1)
P('An AI assistant built this project: it wrote the code, ran the training and evaluation, produced the screenshots, the demo video and this manual (the narration audio was generated with a text-to-speech service), directed by the entrant, who set the goal of real trained models with honest, measured results. Libraries: PyTorch, NumPy, pandas, puppeteer-core for tests and recording, ffmpeg for the video. Dataset: HARTH, Logacjov et al., Sensors 21(23):7853 (2021), UCI ML Repository #779, CC BY 4.0. Code is MIT licensed.')

D.add_heading('13. Glossary',1)
G=[('Accelerometer','Sensor measuring acceleration along three axes.'),('Augmentation','Random changes applied to training data to make a model robust.'),('Calibration','Agreement between stated confidence and actual accuracy.'),('CNN','Convolutional neural network.'),('Confusion matrix','Table of true class against predicted class.'),('Cross-validation','Repeated train/test splits so every sample is tested once.'),('ECE','Expected calibration error.'),('F1','Harmonic mean of precision and recall.'),('Macro-F1','Average of per-class F1, each class weighted equally.'),('Pooled','All out-of-fold predictions combined into one score.'),('Subject-held-out','Test people are never in training.'),('Temperature scaling','Single-parameter confidence calibration.'),('Window','A fixed slice of the signal (here 2 seconds) classified at once.'),('Jerk','Rate of change of acceleration.'),('Logit','Raw model output before softmax.'),('Softmax','Function turning logits into probabilities that sum to 1.')]
T([['Term','Meaning']]+[list(g) for g in G],[1.6,4.9])
D.save('docs/MotionProof_Interview_Manual.docx')
