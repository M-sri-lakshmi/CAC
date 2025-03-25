from flask import *
import os
from werkzeug.utils import secure_filename
import tempfile
from audio import ImprovedAudioSteganography

app = Flask(__name__)
app.secret_key = 'steg_research_secret_key_2025'  # Change this in production
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure key

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure allowed extensions
ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "admin@123":
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid credentials. Please try again.'
    return render_template('login.html', error=error)

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html')
    flash('Please login first', 'danger')
    return redirect(url_for('login'))


@app.route('/encrypt', methods=['GET', 'POST'])
def encrypt():
    if request.method == 'POST':
        # Get message from form
        message = request.form.get('message', '')

        if not message:
            flash('Please enter a message to hide', 'error')
            return redirect(url_for('encrypt'))

        try:
            # Initialize steganography class
            stego = ImprovedAudioSteganography()

            # Generate unique filename
            output_filename = f"hidden_message_{tempfile.mktemp(prefix='', dir='').lstrip('/')}.wav"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

            # Generate audio with hidden message
            stego.embed_message(message, output_path)

            # Provide download link
            return render_template('download.html', filename=output_filename)

        except Exception as e:
            flash(f'Error generating audio: {str(e)}', 'error')
            return redirect(url_for('encrypt'))

    return render_template('encrypt.html')


@app.route('/decrypt', methods=['GET', 'POST'])
def decrypt():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('decrypt'))

        file = request.files['file']

        # Check if file was selected
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('decrypt'))

        # Check if file has allowed extension
        if file and allowed_file(file.filename):
            # Get the expected message length
            message_length = int(request.form.get('message_length', 10))

            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                # Extract message
                stego = ImprovedAudioSteganography()
                extracted_message = stego.extract_message(file_path)

                # Show result
                return render_template('result.html', message=extracted_message)

            except Exception as e:
                flash(f'Error extracting message: {str(e)}', 'error')
                return redirect(url_for('decrypt'))
        else:
            flash('Only .wav files are allowed', 'error')
            return redirect(url_for('decrypt'))

    return render_template('decrypt.html')


@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename),
                     as_attachment=True)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500


# Cleanup function to remove old files
@app.before_request
def cleanup_old_files():
    try:
        # Get list of files in upload folder
        files = os.listdir(app.config['UPLOAD_FOLDER'])

        # Get current time
        current_time = os.time()

        # Remove files older than 1 hour
        for file in files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > 3600:  # 1 hour in seconds
                    os.remove(file_path)
    except Exception:
        pass


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)