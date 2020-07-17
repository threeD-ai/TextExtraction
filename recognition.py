def recognize(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
        global copy2
        copy2 = np.array(img)
        copy = np.array(img)
        img = np.array(img)
        make_clean_folder('./temp/')
        filename, file_ext = os.path.splitext(os.path.basename(img_file))
        excel_file = dirname + filename + '.xlsx'
        res_file = dirname + filename + '_res.txt'
        res_img_file = dirname + filename + '_res.jpg'
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        pageno = '0'
        if filename[-2]=='_':
            pageno = filename[-1]
        elif filename[-3]=='_':
            pageno = filename[-2:] 

        excelbook = xw.Workbook(excel_file)
        excelsheet = excelbook.add_worksheet(pageno)
        excelsheet.set_column(4, 4, 31)
        excelsheet.set_column(5, 5, 25)
        excelsheet.set_column(6, 7, 20)
        excelsheet.set_default_row(15)

        bold = excelbook.add_format({'bold': True})
        excelsheet.write(0,0, 'start_X', bold)
        excelsheet.write(0,1, 'start_Y', bold)
        excelsheet.write(0,2, 'end_X', bold)
        excelsheet.write(0,3, 'end_Y', bold)
        excelsheet.write(0,5, 'Text', bold)
        excelsheet.write(0,4, 'Image', bold)
        excelsheet.write(0,6, 'Ground_Truth', bold)
        excelsheet.write(0,7, 'Label', bold)
        count = 0
        image_list= []
        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)
                poly = poly.reshape(-1, 2)
                if (poly[1][0])-(poly[0][0]) == 0 or (poly[0][1])-(poly[3][1]) == 0:
                    continue
                excelsheet.write(i+1,0,poly[0][0])
                excelsheet.write(i+1,1,poly[0][1])
                excelsheet.write(i+1,2,poly[2][0])
                excelsheet.write(i+1,3,poly[2][1])
                excelsheet.write(i+1,6,'---')
                excelsheet.write(i+1,7,'###')
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)
                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)
                else:
                    image_list.append([i+1,poly[0][1],poly[3][1],poly[0][0],poly[1][0]])
                    roi0 = img[poly[0][1]:poly[3][1], poly[0][0]:poly[1][0],:]
                    try:
                        resized = resize_height(roi0, 20)
                    except:
                        resized = roi0 
                    cv2.imwrite('temp/' +str(i+1) + '.jpg', resized)

                    excelsheet.insert_image(i+1, 4,('temp/' +str(i+1) + '.jpg'), {'x_offset':3, 'y_offset':2, 'object_position':1})
                    cv2.polylines(copy, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)            

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            image_list = np.array(image_list)
            for i, output in executor.map(lambda box: pytesseract.image_to_string(image[box[1]:box[2], box[3]]), image_list):
                excelsheet.write(i, 5, output)

        cv2.imwrite(res_img_file, copy)
        excelbook.close()

