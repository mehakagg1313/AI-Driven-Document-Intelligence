
<head>
    AI-Driven Document Intelligence System
</head>
<h1>AI-Driven Document Intelligence System</h1>

<h2>Overview</h2>
<p>This repository showcases an <strong>AI-Driven Document Intelligence System</strong> designed for smart document categorization, fraud detection, and interactive question answering. The system integrates deep learning models, Optical Character Recognition (OCR), and Large Language Models (LLMs) to streamline document management processes and enable secure, efficient workflows.</p>
<hr>
<h2>Proposed Methodology</h2>

<hr>![methodology](https://github.com/user-attachments/assets/2fd22b2f-d684-4e1d-a0f3-f893c4b3049f)

<h2>Features</h2>
<h3>1. Document Classification</h3>
<p>The document classification module leverages <strong>Custom CNN</strong> and <strong>VGG16</strong> models to categorize documents such as Aadhaar cards, PAN cards, passports, and more.</p>
<ul>
<li><strong>Custom CNN Model:</strong>
            <ul>
                <li>Built from scratch with hyperparameter tuning.</li>
                <li>Features 9 layers, dropout, and ReLU activations.</li>
                <li>Achieves ~83.33% validation accuracy.</li>
            </ul>
        </li>
        <li><strong>VGG16 Model:</strong>
            <ul>
                <li>Utilizes pre-trained VGG16 architecture fine-tuned for document classification.</li>
                <li>Incorporates additional layers for enhanced performance, achieving ~87.69% validation accuracy.</li>
            </ul>
        </li>
    </ul>
<h4>How It Works:</h4>
<ol>
        <li>Pre-process input images.</li>
        <li>Extract features using convolutional layers.</li>
        <li>Classify documents into pre-defined categories.</li>
    </ol>
<h4>Key Results:</h4>
<ul>
        <li>Improved accuracy with data augmentation and pre-trained features.</li>
    </ul>

<hr>

<h3>2. Fraud Detection using Template Matching</h3>
<p>This feature uses template-matching algorithms to detect subtle discrepancies in documents and identify fraudulent ones.</p>

<ul>
        <li><strong>Template Extraction:</strong>
            <ul>
                <li>Converts images to grayscale and enhances contrast.</li>
                <li>Uses edge detection and contour detection to identify Regions of Interest (ROIs).</li>
            </ul>
        </li>
        <li><strong>Template Matching:</strong>
            <ul>
                <li>Matches features using ORB descriptors and a brute-force matcher.</li>
                <li>Evaluates similarity through Structural Similarity Index (SSIM) and nearest neighbor ratio.</li>
            </ul>
        </li>
        <li><strong>Fraud Detection:</strong>
            <ul>
                <li>Calculates similarity and match thresholds to label documents as "genuine" or "potentially fake."</li>
            </ul>
        </li>
    </ul>

<h4>How It Works:</h4>
<ol>
        <li>Extract key features from the document image.</li>
        <li>Compare features against authentic document templates.</li>
        <li>Classify documents based on similarity scores.</li>
    </ol>

<h4>Key Results:</h4>
<ul>
        <li>High accuracy in detecting fraud, including subtle variations like logo differences or font size changes.</li>
    </ul>

<hr>

<h3>3. Optical Character Recognition (OCR)</h3>
<p>The OCR module uses <strong>Pytesseract</strong> to extract textual information from scanned documents.</p>

<h4>Preprocessing:</h4>
<ul>
        <li>Images are resized, sharpened, and converted to grayscale.</li>
        <li>Dynamic thresholding is applied to enhance text clarity.</li>
    </ul>

<h4>Text Extraction:</h4>
<ul>
        <li>Extracted text is refined using document-specific keywords.</li>
        <li>Stores processed data for further analysis.</li>
    </ul>

<h4>Key Results:</h4>
<ul>
        <li>Improved text retrieval accuracy with advanced preprocessing techniques.</li>
    </ul>

<hr>

<h3>4. Interactive Question Answering</h3>
![qna](https://github.com/user-attachments/assets/3001aa26-6e6c-4f19-a276-7ce39e5236f5)

<p>This feature enables users to query documents directly using <strong>LLMs</strong> integrated with the <strong>LangChain</strong> framework.</p>

<h4>Workflow:</h4>
<ul>
        <li>Converts PDF documents into readable text.</li>
        <li>Splits text into manageable chunks using CharacterTextSplitter.</li>
        <li>Generates vector embeddings for semantic analysis using OpenAI’s GPT models.</li>
    </ul>

<h4>Cosine Similarity:</h4>
<p>Matches user queries with document content for precise information retrieval.</p>

<h4>Key Results:</h4>
<ul>
        <li>Achieved a confidence score of 98.43% for single-page PDFs, outperforming traditional BERT models.</li>
    </ul>

<hr>

<h2>Dataset</h2>
<p>The dataset includes scanned images and PDFs of documents such as Aadhaar cards, PAN cards, and more. Data was augmented through flipping, rotation, and translation to enhance generalization.</p>

<table border="1">
        <tr>
            <th>Document Type</th>
            <th>No. of Documents</th>
        </tr>
        <tr>
            <td>Aadhaar Card</td>
            <td>129</td>
        </tr>
        <tr>
            <td>PAN Card</td>
            <td>45</td>
        </tr>
        <tr>
            <td>Driving License</td>
            <td>64</td>
        </tr>
        <tr>
            <td>Voter ID</td>
            <td>76</td>
        </tr>
        <tr>
            <td>Passport</td>
            <td>37</td>
        </tr>
    </table>

<hr>

<h2>Performance Metrics</h2>
<p>Key metrics used to evaluate the system:</p>
<ul>
        <li><strong>Train Accuracy:</strong> Measures model performance on training data.</li>
        <li><strong>Validation Accuracy:</strong> Reflects generalization on unseen data.</li>
        <li><strong>SSIM Score:</strong> Measures similarity between images.</li>
        <li><strong>Confidence Score:</strong> Evaluates the reliability of question-answering models.</li>
    </ul>

<table border="1">
        <tr>
            <th>Model</th>
            <th>Train Accuracy (%)</th>
            <th>Validation Accuracy (%)</th>
        </tr>
        <tr>
            <td>CNN</td>
            <td>69.44</td>
            <td>83.33</td>
        </tr>
        <tr>
            <td>VGG16</td>
            <td>90.51</td>
            <td>87.69</td>
        </tr>
    </table>

<hr>

<h2>Tools and Technologies</h2>
    <ul>
        <li><strong>Deep Learning Models:</strong> Custom CNN, VGG16.</li>
        <li><strong>OCR:</strong> Pytesseract.</li>
        <li><strong>LLMs:</strong> OpenAI GPT models with LangChain.</li>
        <li><strong>Development Platform:</strong> Google Colab.</li>
        <li><strong>Libraries:</strong> TensorFlow, PyTorch, OpenCV, NumPy, Pandas.</li>
    </ul>

<hr>

<h2>Installation and Setup</h2>
<ol>
        <li>Clone this repository.</li>
        <li>Install dependencies:
            <pre><code>pip install -r requirements.txt</code></pre>
        </li>
        <li>Run the main scripts for each feature:
            <ul>
                <li><strong>Document Classification:</strong> <code>python classify_documents.py</code></li>
                <li><strong>Fraud Detection:</strong> <code>python fraud_detection.py</code></li>
                <li><strong>OCR:</strong> <code>python extract_text.py</code></li>
                <li><strong>Interactive QA:</strong> <code>python interactive_qa.py</code></li>
            </ul>
        </li>
    </ol>

<hr>

<h2>Contribution</h2>
<p>Feel free to contribute by opening a pull request or reporting issues. Your suggestions and feedback are valuable to us.</p>

