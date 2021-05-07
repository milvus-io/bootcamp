import argparse
import numpy as np
import omdb
import json


def get_movies_info(movie_data):
    with open(movie_data.replace('movies.dat', 'movies_info.csv'), 'a') as f_w:
        with open(movie_data, 'r') as f_r:
            count = 0
            genres = ['Animation', "Children's", 'Comedy', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Sci-Fi', 'Documentary', 'War', 'Musical', 'Mystery', 'Film-Noir', 'Western']
            for line in f_r.readlines():
                count += 1
                if count<=2998:
                    continue
                
                line = line.split("::")
                title = line[1]
                year = title[-5:-1]
                title = title[:-7]

                genre = line[2].strip().split("|")
                for g in genre:
                    if g not in genres:
                        genres.append(g)
                print("----------genres:", genres)

                res = omdb.request(t=title, y=year, r='json', apikey='432549e')
                content = res.content
                line = line[0] + ',' + str(content).replace('b\'','').replace('\'','')
                print(line)
                f_w.write(line+'\n')


def update_info(movie_data, info_data):
    movies = {}
    with open(movie_data, 'r') as f_r:
        for line in f_r.readlines():
            line = line.split("::")
            title = line[1]
            year = title[-5:-1]
            title = title[:-7]
            genre = line[2].strip().split("|")
            movies[line[0]] = [title, year, genre]
        print(movies)

    with open(info_data.replace('movies_info.csv', 'movies_update.csv'), 'a+') as f_w:
        with open(info_data, 'r') as f_r:
            count = 0
            for line in f_r.readlines():
                count += 1
                if count <= 2934:
                    continue
                if "Movie not found!" in line:
                    info = line.split(",")
                    num = info[0]
                    if "The" in movies[num][0]:
                        print("---------The in title!")
                        movies[num][0] = movies[num][0].replace(", The", "")
                        movies[num][0] = "The " + movies[num][0]
                    print("movies[num]", movies[num])

                    title = movies[num][0]
                    year = movies[num][1]
                    res = omdb.request(t=title, y=year, r='json', apikey='dc870354')
                    content = res.content
                    content = str(content).replace('b\'','').replace('\'','')+'\n'
                    if "Movie not found!" in content:
                        content = "{\"Title\":\"" + str(movies[num][0])+"\",\"Year\":\""+str(movies[num][1])+"\",\"Genre\":\""+" ".join(movies[num][2])+"\",\"Posters\":\""+"N/A\"}"+"\n"
                    line = str(num) + ',' + content
                    print(line)
                f_w.write(line)


def get_posters(info_data):
    with open(info_data.replace('movie_id.dat', 'movies_posters.sh'), 'a+') as f_w:
        with open(info_data, 'r') as f_r:
            for line in f_r.readlines():
                line = line.strip().split("::")
                print("-------line:", line)
                num = line[0]
                info = line[1]
                try:
                    info = json.loads(info.replace("\\", ""))
                    print("-------info:", info)
                except:
                    info = json.loads(info.replace("\\\"", "").replace("\\", ""))
                    print("-------except info:", info)

                poster = info["Poster"]
                if not "N/A" in poster:
                    line = "wget -c "+ poster +' -O '+ num +'.jpg\n'
                    print(line)
                    f_w.write(line)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('movie_data', type=str)
    parser.add_argument('info_data', type=str)
    args = parser.parse_args()

    # update_info(args.movie_data, args.info_data)
    # get_movies_info(args.movie_data)
    get_posters(args.info_data)


if __name__ == '__main__':
    main()
