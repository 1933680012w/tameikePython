import mysql.connector

dbpassword = 'dbpassword'

def sqlSelect(sqlExecuteSentence):
    conn = mysql.connector.connect(user='tameike',password=dbpassword,host='localhost',database='tameike_db')
    cur=conn.cursor()

    try:
        cur.execute(sqlExecuteSentence)
    except:
        conn.rollback()
        raise

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def sqlInsertUpdate(sqlExecuteSentence):
    conn = mysql.connector.connect(user='tameike',password=dbpassword,host='localhost',database='tameike_db')
    cur=conn.cursor()
    try:
        cur.execute(sqlExecuteSentence)
        cur.close()
        conn.commit()
        conn.close()
    except:
        conn.rollback()
        raise