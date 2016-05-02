echo "NB_FACE_TEST"
python naive_bayes.py facedata/facedatatrain facedata/facedatatrainlabels facedata/facedatatest facedata/facedatatestlabels > NB_Face_Test.dat
echo "NB_FACE_VALID"
python naive_bayes.py facedata/facedatatrain facedata/facedatatrainlabels facedata/facedatavalidation facedata/facedatavalidationlabels > NB_Face_Valid.dat
echo "NB_DIGIT_TEST"
python naive_bayes.py digitdata/trainingimages digitdata/traininglabels digitdata/testimages digitdata/testlabels > NB_Digit_Test.dat
echo "NB_DIGIT_VALID"
python naive_bayes.py digitdata/trainingimages digitdata/traininglabels digitdata/validationimages digitdata/validationlabels > NB_Digit_Valid.dat
