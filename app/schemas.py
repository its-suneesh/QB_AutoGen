# app/schemas.py

from marshmallow import Schema, fields, validate

class LoginSchema(Schema):
    username = fields.Str(required=True)
    password = fields.Str(required=True)

class BookDetailsSchema(Schema):
    BookName = fields.Str(required=True)
    BookType = fields.Str(required=True)

class RuleSchema(Schema):
    questionId = fields.Int(required=True)
    questionType = fields.Str(required=True)
    difficultyLevel = fields.Str(required=True)
    cognitiveLevel = fields.Str(required=True)
    mark = fields.Int(required=True)
    numberOfQuestions = fields.Int(required=True, validate=lambda n: n > 0)

class GenerateSchema(Schema):
    module = fields.Str(required=True)
    content = fields.Str(required=True)
    Rules = fields.List(fields.Nested(RuleSchema), required=True)
    BookDetails = fields.List(fields.Nested(BookDetailsSchema), required=True)
    model = fields.Str(required=True)