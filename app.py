import pyrebase, random, os
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for
from werkzeug import secure_filename
from ocr import performOcr
app = Flask(__name__)
config = {
  "apiKey": "AIzaSyD_lqIgTFSItpDOGZ1NycLoB6w4nrXWL2I",
  "authDomain": "guvi-7679a.firebaseapp.com",
  "databaseURL": "https://guvi-7679a.firebaseio.com",
  "storageBucket": "guvi-7679a.appspot.com",
  "serviceAccount": "credentials/serviceAccountCredentials.json"
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()
key=""
#Initialze person as dictionary
person = {"is_logged_in": False, "name": "", "email": "", "uid": ""}




@app.route("/")
def login():
    result = ""
    return render_template("login.html", result = result)

#Sign up/ Register
@app.route("/signup")
def signup():
    return render_template("signup.html")

#Welcome page
@app.route("/welcome")
def welcome():
    if person["is_logged_in"] == True:
        return render_template("welcome.html", email = person["email"], name = person["name"])
    else:
        return redirect(url_for('login'))

@app.route("/login", methods = ["POST", "GET"])
def result():
    if request.method == "POST":        #Only if data has been posted
        result = request.form           #Get the data
        email = result["email"]
        password = result["pass"]
        try:
            #Try signing in the user with the given information
            user = auth.sign_in_with_email_and_password(email, password)
            global person
            person["is_logged_in"] = True
            person["email"] = user["email"]
            person["uid"] = user["localId"]
            data = db.child("users").get()
            person["name"] = user["email"]
            return render_template("welcome.html")
        except Exception as e:
            result = "Incorrect email or password"
            return render_template("login.html", result = result)
    else:
        if person["is_logged_in"] == True:
            return render_template("welcome.html", email = person["email"], name = person["name"])
        else:
            result = ""
            render_template("login.html", result = result)

#If someone clicks on register, they are redirected to /register
@app.route("/register", methods = ["POST", "GET"])
def register():
    if request.method == "POST":        #Only listen to POST
        result = request.form
        #Get the data submitted
        email = result["email"]
        password = result["pass"]
        name = result["name"]
        try:
            #Try creating the user account using the provided data
            auth.create_user_with_email_and_password(email, password)
            #Login the user
            user = auth.sign_in_with_email_and_password(email, password)
            #Add data to global person
            global person
            person["is_logged_in"] = True
            person["email"] = user["email"]
            person["uid"] = user["localId"]
            person["name"] = name
            #Append data to the firebase realtime database
            data = {"name": name, "email": email}
            db.child("users").child(person["uid"]).set(data)
            #Go to welcome page
            return redirect(url_for('welcome'))
        except Exception as e:
            result = "Incorrect credentials"
            #If there is any error, redirect to register
            return render_template("signup.html", result = result)

    else:
        if person["is_logged_in"] == True:
            return redirect(url_for('welcome'))
        else:
            result = ""
            return render_template("signup.html", result = result)


@app.route("/alphabets", methods = ["POST", "GET"])
def Alphabets():
    file = random.choice([
        x for x in os.listdir(os.path.join('static', 'dataset', 'alphabets'))
        if os.path.isfile(os.path.join('static', 'dataset', 'alphabets', x))
    ])
    print(file,"fhgfh")
    #global key
    file1=file.split(".")[0]
    set_key(file1)
    print(file)
    file = os.path.join('static', 'dataset', 'alphabets', file)
    return render_template("flashcard.html", file = file)

@app.route('/result', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join('static', 'uploads', filename))
        result = performOcr(os.path.join('static', 'uploads', filename))
        #global key
        set=False
        for i in get_key():
            if i in result:
                set=True
                print(result)
                return render_template('happy.html')
        #resultTwo = ""
        #if resultTwo in result:
        return render_template('Sad.html')
def set_key(val):
    global key
    key=val
    print(key,"set")
def get_key():
    global key
    print("inside",key)
    return key


if __name__ == "__main__":
    app.run(debug=True)
