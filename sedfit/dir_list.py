import os
__all__ = ["root_path", "filter_path", "template_path"]

#-> Obtain the current path
pathList = os.path.abspath(__file__).split("/")
#-> Create the path to the root path
root_path = "/".join(pathList[0:-2]) + "/"
#-> Create the path to the filters
pathList[-1] = "filters/"
filter_path = "/".join(pathList)
#-> Create the path to the templates
pathList[-2] = "template/"
template_path = "/".join(pathList[0:-1])

if __name__ == "__main__":
    print root_path
