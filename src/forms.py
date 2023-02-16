from flask_wtf import FlaskForm
from wtforms import (
    Form,
    BooleanField,
    StringField,
    PasswordField,
    SubmitField,
    FloatField,
    IntegerField,
)
from wtforms.validators import DataRequired, Length, Email, EqualTo


class PatientForm(FlaskForm):
    DBP1 = FloatField("DBP1", validators=[DataRequired()])
    SBP1 = IntegerField("SBP1", validators=[DataRequired()])
    # V1_BMI = FloatField("V1_BMI", validators=[DataRequired()])
    V1_FHISTORY_DM = IntegerField("V1_FHISTORY_DM", validators=[DataRequired()])
    V1_FHISTORY_HTN = IntegerField("V1_FHISTORY_HTN", validators=[DataRequired()])
    V1_UPRO = IntegerField("V1_UPRO", validators=[DataRequired()])
    V1_USUG = IntegerField("V1_USUG", validators=[DataRequired()])
    V1_glu_amt = FloatField("V1_glu_amt", validators=[DataRequired()])
    V1_preg_dm = IntegerField("V1_preg_dm", validators=[DataRequired()])
    aft_cal = IntegerField("aft_cal", validators=[DataRequired()])
    aft_sleep = FloatField("aft_sleep", validators=[DataRequired()])
    age = IntegerField("age", validators=[DataRequired()])
    bef_cont = IntegerField("bef_cont", validators=[DataRequired()])
    bef_folt = IntegerField("bef_folt", validators=[DataRequired()])
    bef_ink = IntegerField("bef_ink", validators=[DataRequired()])
    bef_sleep = FloatField("bef_sleep", validators=[DataRequired()])
    bmi_aft = FloatField("bmi_aft", validators=[DataRequired()])
    childbirth_no = IntegerField("childbirth_no", validators=[DataRequired()])
    gedu = IntegerField("gedu", validators=[DataRequired()])
    goccp = IntegerField("goccp", validators=[DataRequired()])
    met_ope = IntegerField("met_ope", validators=[DataRequired()])
    moccp_t = FloatField("moccp_t", validators=[DataRequired()])
    pre_birth = IntegerField("pre_birth", validators=[DataRequired()])
    preg_htn2 = IntegerField("preg_htn2", validators=[DataRequired()])
    # v1_medu = IntegerField("v1_medu", validators=[DataRequired()])
    submit = SubmitField("결과확인")
