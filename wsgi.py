from app import create_app
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
application = create_app()
app = application

if __name__ == "__main__":
    app.run()
