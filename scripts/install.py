#!/usr/bin/env python3
"""
Installation and setup script for PulseByte.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python 3.8+ required, but found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_venv():
    """Create virtual environment."""
    venv_path = Path("venv")
    if venv_path.exists():
        print("✓ Virtual environment already exists")
        return True
    
    return run_command("python3 -m venv venv", "Creating virtual environment")


def install_requirements():
    """Install Python requirements."""
    # Determine the activation script path based on OS
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_command = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_script = "venv/bin/activate"
        pip_command = "venv/bin/pip"
    
    # Install requirements
    return run_command(f"{pip_command} install -r requirements.txt", "Installing Python packages")


def download_nltk_data():
    """Download required NLTK data."""
    nltk_command = """
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('brown', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'Error downloading NLTK data: {e}')
"
"""
    return run_command(nltk_command, "Downloading NLTK data")


def download_textblob_corpora():
    """Download TextBlob corpora."""
    textblob_command = "python3 -m textblob.download_corpora"
    return run_command(textblob_command, "Downloading TextBlob corpora")


def create_env_file():
    """Create .env file from template."""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if env_file.exists():
        print("✓ .env file already exists")
        return True
    
    if env_example.exists():
        try:
            env_file.write_text(env_example.read_text())
            print("✓ Created .env file from template")
            print("  Remember to add your API keys to the .env file!")
            return True
        except Exception as e:
            print(f"✗ Error creating .env file: {e}")
            return False
    else:
        print("✗ env.example file not found")
        return False


def test_installation():
    """Test the installation."""
    test_command = "python3 -c \"from src.models import Article; print('✓ Import test passed')\""
    return run_command(test_command, "Testing installation")


def main():
    """Main installation process."""
    print("🚀 Setting up PulseByte...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_venv():
        print("Failed to create virtual environment. You may need to install it manually.")
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Download ML model data
    print("\nDownloading ML model data...")
    download_nltk_data()
    download_textblob_corpora()
    
    # Test installation
    if test_installation():
        print("\n🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Activate the virtual environment:")
        if os.name == 'nt':
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("2. Add your API keys to the .env file")
        print("3. (Optional) Set up PostgreSQL: python3 scripts/setup_database.py --install-help")
        print("4. Test the installation: python3 main.py test")
        print("5. Run your first scraping: python3 main.py run --keywords 'technology'")
    else:
        print("\n❌ Installation completed with errors.")
        print("Please check the error messages above and try again.")


if __name__ == "__main__":
    main()
