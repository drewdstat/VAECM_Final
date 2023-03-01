def getarchi(nlayers=3,D=15,OUT=4,C1=15,C2=60,C3=20):
    if(nlayers==1):
        ARCHI = ([('input',D),
            ('dense', (C1, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (OUT, 'glorot_uniform', 'glorot_normal') ),
        ],[('input' , OUT),
            ('dense', (C1, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (D, 'glorot_uniform', 'glorot_normal') ),
        ])
    elif(nlayers==2):
        ARCHI = ([('input',D),
            ('dense', (C1, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (C2, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (OUT, 'glorot_uniform', 'glorot_normal') ),
        ],[('input' , OUT),
            ('dense', (C2, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (C1, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (D, 'glorot_uniform', 'glorot_normal') ),
        ])
    elif(nlayers==3):
        ARCHI = ([('input',D),
            ('dense', (C1, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (C2, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (C3, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (OUT, 'glorot_uniform', 'glorot_normal') ),
        ],[('input' , OUT),
            ('dense', (C3, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (C2, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (C1, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (D, 'glorot_uniform', 'glorot_normal') ),
        ])
    else: raise ValueError('nlayers must be 3 or fewer, not ' + str(nlayers))
    
    return ARCHI
