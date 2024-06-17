import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request
from flask import send_file
from reportlab.lib.pagesizes import portrait, A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest


summary_for_pdf = []

app = Flask(__name__)

@app.route('/')
def start():
    return render_template('start.html')
@app.route('/start')
def star1():
    return render_template('start.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/under_construct')
def und():
    return render_template('under_constr.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/index')
def index():
    # Remove all files from the static/plots folder
    folder_path = 'static/plots'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    return render_template('form-cs.html')

@app.route('/clear_plots')
def clear_plots():
    folder_path = 'static/plots'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    return '', 204

@app.route('/readme', methods=['GET', 'POST'])
def readme():
    return render_template('readme.html')


ALLOWED_EXTENSIONS = {'csv', 'txt', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Route for generating and displaying results




def get_plot_filenames():
    plot_dir = 'static/plots'
    plot_filenames = [filename for filename in os.listdir(plot_dir) if filename.endswith('.png')]
    return plot_filenames

# Helper function to get summary of the plot and outlier data
def get_plot_summary(dataset, feature_names, outliers):
    summary = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            feature1 = feature_names[i]
            feature2 = feature_names[j]
            plot_summary = f"Plot: {feature1} vs {feature2}\n"
            plot_summary += "Description:\n"
            plot_summary += "- Scatter plot of the data points\n"
            plot_summary += "- Linear regression line\n"
            plot_summary += "- Outlier points in red\n"
            plot_summary += "Outlier Data:\n"
            outlier_data = dataset[outliers]
            outlier_data_summary = f"Number of outliers: {len(outlier_data)}\n"
            outlier_data_summary += f"{outlier_data.describe()}\n"
            plot_summary += outlier_data_summary
            summary_for_pdf.append(plot_summary)

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            feature1 = feature_names[i]
            feature2 = feature_names[j]
            plot_summary = f"<h3>Plot: {feature1} vs {feature2}</h3>"
            plot_summary += "<hr>"
            plot_summary += "<h4>Description:</h4>"
            plot_summary += "<ul>"
            plot_summary += "<li>Scatter plot of the data points</li>"
            plot_summary += "<li>Linear regression line</li>"
            plot_summary += "<li>Outlier points in red</li>"
            plot_summary += "</ul>"
            plot_summary += "<hr>"
            plot_summary += "<h4>Outlier Data:</h4>"
            outlier_data = dataset[outliers]
            plot_summary += f"<p>Number of outliers: {len(outlier_data)}</p>"
            plot_summary += f"<pre>{outlier_data.describe().to_string()}</pre>"
            summary.append(plot_summary)
    return summary

@app.route('/generate-linear-plots', methods=['GET', 'POST'])
def generate_linear_plots():
    if request.method == 'POST':
        # Get form data
        print("Getting Form Data")
        algo_type = request.form['algo_type']
        dataset_selection = request.form['dataset']
        print("Dataset received")

        if request.form['dataset'] == 'custom':
            print("Dataset custom")
            file = request.files['custom_dataset']
            if file and allowed_file(file.filename):
                # Save the custom dataset file to the "datasets" folder
                if file:
                    filename = 'custom.csv'  # Set the desired filename
                    file_path = os.path.join('datasets', filename)
                    file.save(file_path)  # Save the file
                    # Perform regression on the custom dataset

        # Determine dataset selection and load dataset
        if dataset_selection == 'dataset1':
            dataset = pd.read_csv('datasets/data.csv')
        elif dataset_selection == 'dataset2':
            dataset = pd.read_csv('datasets/fundamentals.csv')
        elif dataset_selection == 'dataset3':
            dataset = pd.read_csv('datasets/kc_house_data.csv')
        elif dataset_selection == 'dataset4':
            dataset = pd.read_csv('datasets/Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv')
        elif dataset_selection == 'dataset5':
            dataset = pd.read_csv('datasets/placement.csv')
        elif dataset_selection == 'dataset6':
            dataset = pd.read_csv('datasets/population.csv')
        elif dataset_selection == 'dataset7':
            dataset = pd.read_csv('datasets/prices.csv')
        elif dataset_selection == 'dataset8':
            dataset = pd.read_csv('datasets/prices-split-adjusted.csv')
        elif dataset_selection == 'dataset9':
            dataset = pd.read_csv('datasets/Real estate.csv')
        elif dataset_selection == 'dataset10':
            dataset = pd.read_csv('datasets/Salary_dataset.csv')
        elif dataset_selection == 'dataset11':
            dataset = pd.read_csv('datasets/Salary_Data.csv')
        elif dataset_selection == 'dataset12':
            dataset = pd.read_csv('datasets/securities.csv')
        elif dataset_selection == 'dataset13':
            dataset = pd.read_csv('datasets/who_dataset.csv')
        elif dataset_selection == 'dataset14':
            dataset = pd.read_csv('datasets/winequality-red.csv')
        elif dataset_selection == 'custom':
            dataset = pd.read_csv('datasets/custom.csv')
        else:
            print("ASSIGNING DATASET ERROR")
            return render_template('error.html')

        # Perform linear regression
        if algo_type == 'linear_regression':
            X = dataset.iloc[:, 0].values.reshape(-1, 1)  # First column as the independent variable
            y = dataset.iloc[:, 1].values.astype(float)  # Convert target variable to float

            # Pass the dependent and independent variable names to the template
            dependent_variable = dataset.columns[1]  # Assuming the dependent variable is the second column
            independent_variable = dataset.columns[0]  # Assuming the independent variable is the first column

            if len(X) > 0 and len(y) > 0:  # Check if dataset has at least one sample
                feature_names = dataset.columns.tolist()[:2]  # Select first two columns as default

                if len(dataset.columns) < 2:
                    error_message = "Dataset does not have enough columns. Select at least 2 columns."
                    return render_template('error.html', error_message=error_message)

                # Convert dataset to HTML table
                table_html = dataset.to_html(index=False)

                # Fit linear regression model
                regressor = LinearRegression()
                regressor.fit(dataset[feature_names], y)

                # Perform anomaly detection
                anomaly_detector = IsolationForest(contamination=0.1)  # Adjust contamination level if needed
                anomaly_detector.fit(dataset[feature_names])
                outliers = anomaly_detector.predict(dataset[feature_names]) == -1
                bg_color = request.form.get('bg_color')
                reg_line_color = request.form.get('reg_line_color')
                outlier_color = request.form.get('outlier_color')

                # Pass the colors to the plot generation function

                # Generate scatter plots for selected features
                plot_dir = os.path.join('static', 'plots')
                plot_dir = os.path.join('static', 'plots')
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)

                for i in range(len(feature_names)):
                    for j in range(i + 1, len(feature_names)):
                        fig, ax = plt.subplots()
                        ax.scatter(dataset[feature_names[i]], dataset[feature_names[j]], c=bg_color)
                        ax.scatter(dataset[feature_names[i]][outliers], dataset[feature_names[j]][outliers],
                                   c=outlier_color)
                        ax.plot(dataset[feature_names[i]], regressor.predict(dataset[feature_names].values),
                                color=reg_line_color)
                        ax.set_xlabel(feature_names[i])
                        ax.set_ylabel(feature_names[j])
                        ax.legend(['Linear Regression', 'Normal', 'Outlier'])
                        ax.set_title('Linear Regression with Outlier Detection')
                        # Save plot image to file
                        plot_file = f'plot_{feature_names[i]}_{feature_names[j]}.png'
                        plot_path = os.path.join(plot_dir, plot_file)
                        plt.savefig(plot_path, format='png')
                        plt.close(fig)

                # Get list of generated plot filenames
                image_names = get_plot_filenames()
                image_paths = [os.path.join('static', 'plots', name) for name in image_names]


                print(image_paths)
                print(image_names)
                # Get plot summary and outlier data
                plot_summary = get_plot_summary(dataset, feature_names, outliers)

                print(plot_summary)

                # Return results HTML
                return render_template('result.html', image_names=image_names, image_paths=image_paths,
                                       dependent_variable=dependent_variable, independent_variable=independent_variable,
                                       table_html=table_html, plot_summary=plot_summary)

            else:
                error_message = 'Dataset does not have enough samples.'
                return render_template('error.html', error_message=error_message)
        else:
            error_message = 'Invalid algorithm type.'
            return render_template('error.html', error_message=error_message)

    return render_template('error.html')



@app.route('/download-pdf')
def download_pdf():
    image_folder = 'static/plots'
    pdf_path = 'static/pdf/special_project_report.pdf'

    # Get all the image files in the folder
    image_files = [filename for filename in os.listdir(image_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]

    # Calculate the number of pages needed
    images_per_page = 2
    total_pages = (len(image_files) + images_per_page - 1) // images_per_page

    # Create a new PDF canvas
    pdf_canvas = canvas.Canvas(pdf_path, pagesize=portrait(A4), bottomup=True)

    # Set margins
    left_margin = 0.75 * inch
    right_margin = A4[0] - 0.75 * inch
    top_margin = A4[1] - 0.75 * inch
    bottom_margin = 0.75 * inch

    # Set background color
    pdf_canvas.setFillColorRGB(0.9, 0.9, 0.9)

    # Add introduction page
    intro_image_path = 'static/intro.png'
    pdf_canvas.drawImage(intro_image_path, x=0, y=0, width=A4[0], height=A4[1])



    # Iterate through the image files and add them to the PDF
    page_count = 0
    for i, image_file in enumerate(image_files):
        # Add a new page if necessary
        if i % images_per_page == 0:
            pdf_canvas.showPage()
            page_count += 1

            # Add header
            header_text = f"Page {page_count}/{total_pages}"
            pdf_canvas.setFont('Helvetica-Bold', 12)
            pdf_canvas.drawCentredString(A4[0] / 2, top_margin - inch, header_text)

        # Calculate the position for the current image
        x = left_margin
        y = top_margin - (i % images_per_page) * (A4[1] / 2) - 4.8 * inch

        # Add the image to the current page
        image_path = os.path.join(image_folder, image_file)
        pdf_canvas.drawImage(image_path, x=x, y=y, width=6.4 * inch, height=4.8 * inch)

        # Add the image name below the image
        pdf_canvas.setFont('Helvetica', 10)
        pdf_canvas.drawRightString(x + 6.3 * inch, y - 0.2 * inch, image_file)

    # Add the summary below the image on the second page
    if len(image_files) == 1:
        pdf_canvas.showPage()
        pdf_canvas.setFont('Helvetica-Bold', 12)
        pdf_canvas.drawString(left_margin, top_margin - inch, 'Report Summary:')
        y_summary = top_margin - inch - 0.5 * inch
        pdf_canvas.setFont('Helvetica-Bold', 10)
        pdf_canvas.setFillColorRGB(0, 0, 0)  # Set font color to black
        for summary in summary_for_pdf:
            lines = summary.split('\n')
            for line in lines:
                pdf_canvas.drawString(left_margin, y_summary, line)
                y_summary -= 0.2 * inch

    # Add ending page
    ending_image_path = 'static/end.png'
    pdf_canvas.showPage()
    pdf_canvas.drawImage(ending_image_path, x=0, y=0, width=A4[0], height=A4[1])

    # Save the PDF file
    pdf_canvas.save()

    # Send the generated PDF file as an attachment
    return send_file(pdf_path, as_attachment=True)




if __name__ == '__main__':
    app.run()
