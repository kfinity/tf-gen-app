# SYS 6016 - Codeathon 3 - "There's an app for that"

This small project is a weekly assignment for a Deep Learning class in my master's program at [UVA School of Data Science](https://datascience.virginia.edu/).

Currently deployed at https://sys6016codeathon3.uk.r.appspot.com/

There's several components of this assignment:

  1. names.csv - a collection of U.S. baby names collected from [the SSA's data](https://www.ssa.gov/oact/babynames/limits.html) (up to 2017)
  2. text_generation_demo.ipynb - working development notebook, a hot mess. How the sausage got made.
  3. namegen/ - the trained TensorFlow SavedModel, currently deployed to Google Cloud AI Platform
  4. namegen-app/ - the Flask app, currently deployed to Google Cloud App Engine
  5. input_a.json - an example input file, corresponding to just the letter "a"
